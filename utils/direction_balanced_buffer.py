"""已知完成时间与固定离线方向组的探索性纯选择器；不采样、不训练。

迁自 Weak 冻结快照，保留原调度、队列因果约束及诊断字段。
这不是无需预知完成时间或总体方向的可部署调度策略。
"""

from collections.abc import Mapping
from itertools import combinations
import math
from numbers import Integral, Real

BUFFER_SIZE = 10
CANDIDATE_LIMIT = 15
BALANCE_EVERY = 5
MAX_REPLACEMENTS = 2
MAX_DEFERRED_ROUNDS = 2


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} 必须为不小于 {minimum} 的整数')
    return int(value)


def _real(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name} 必须为有限实数')
    value = float(value)
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f'{name} 必须有限且不小于 {minimum}')
    return value


def _values(values, name):
    if isinstance(values, (str, bytes, Mapping, set, frozenset)):
        raise ValueError(f'{name} 必须为有序序列')
    try:
        return tuple(values)
    except TypeError as exc:
        raise ValueError(f'{name} 必须为有序序列') from exc


def _previous_ready(state, round_number, clock, costs, stamps):
    if state is None:
        return {}
    if not isinstance(state, Mapping):
        raise ValueError('state 必须为空或为上轮返回状态')
    if not state:
        return {}
    if set(state) != {'last_round', 'clock', 'ready'}:
        raise ValueError('state 必须为空或为上轮返回状态')
    last_round = _integer(state['last_round'], 'state.last_round', 1)
    old_clock = _real(state['clock'], 'state.clock', 0)
    if round_number != last_round + 1 or clock < old_clock:
        raise ValueError('state 要求连续轮次和不递减时钟')
    if not isinstance(state['ready'], Mapping):
        raise ValueError('state.ready 必须为映射')
    ready = {}
    fields = {'completion_time', 'stamp', 'ready_since_round', 'ready_since_time'}
    for client, item in state['ready'].items():
        client = _integer(client, 'state.ready client id')
        if client >= len(costs) or not isinstance(item, Mapping) or set(item) != fields:
            raise ValueError('无效 state.ready 条目')
        completion = _real(item['completion_time'], 'completion_time', 0)
        stamp = _integer(item['stamp'], 'state stamp')
        since_round = _integer(item['ready_since_round'], 'ready_since_round', 1)
        since_time = _real(item['ready_since_time'], 'ready_since_time', 0)
        if (completion <= 0 or completion > old_clock or stamp >= since_round
                or since_round > last_round or since_time != completion):
            raise ValueError('state.ready 条目违反到达因果约束')
        if costs[client] == completion and stamps[client] == stamp:
            ready[client] = dict(completion_time=completion, stamp=stamp,
                                 ready_since_round=since_round, ready_since_time=since_time)
    return ready


def select_buffer(costs, stamps, group_ids, population_weights, round_number,
                  clock, policy='fifo', state=None):
    """无突变、无 RNG 地返回 (record, new_state)。

    正成本为绝对完成时间，非正为不活跃；时间戳为启动版本。
    Weak 每五轮在最早十五个中保留第十五个锚点，至多替换两个 FIFO 项，
    依次最小化 Phi、陈旧度和、时钟增量及完成时间/ID 字典序。
    已完成但等候两轮的更新强制服用；冲突时明确记录回退与越限。
    父调度器必须通过真实选择 hook 消费 selected_ids；仅钳制成本不够。
    """
    round_number = _integer(round_number, 'round_number', 1)
    clock = _real(clock, 'clock', 0)
    if not isinstance(policy, str) or policy not in ('fifo', 'weak'):
        raise ValueError('policy 必须为 fifo 或 weak')
    costs = tuple(_real(c, 'cost') for c in _values(costs, 'costs'))
    stamps = tuple(_integer(s, 'stamp', -1) for s in _values(stamps, 'stamps'))
    groups = tuple(_integer(g, 'group id') for g in _values(group_ids, 'group_ids'))
    weights = tuple(_real(p, 'population weight', 0)
                    for p in _values(population_weights, 'population_weights'))
    if not len(costs) == len(stamps) == len(groups):
        raise ValueError('costs、stamps、group_ids 长度必须相同')
    if (not weights or any(p > 1 for p in weights)
            or not math.isclose(math.fsum(weights), 1.0, rel_tol=0, abs_tol=1e-12)):
        raise ValueError('population_weights 之和必须为 1')
    if any(g >= len(weights) for g in groups):
        raise ValueError('group id 超出 population_weights 范围')
    order_key = lambda client: (costs[client], client)
    active = sorted((i for i, c in enumerate(costs) if c > 0), key=order_key)
    if len(active) < BUFFER_SIZE:
        raise ValueError('至少需要十个活跃更新')
    if any(stamps[i] < 0 or stamps[i] >= round_number for i in active):
        raise ValueError('活跃时间戳必须位于 [0, round_number-1]')
    previous = _previous_ready(state, round_number, clock, costs, stamps)
    baseline = active[:BUFFER_SIZE]
    baseline_set = set(baseline)
    fifo_clock = max(clock, costs[baseline[-1]])
    balance_round = policy == 'weak' and round_number % BALANCE_EVERY == 0
    candidates = active[:CANDIDATE_LIMIT] if balance_round else list(baseline)
    anchor = candidates[-1] if balance_round and len(candidates) == CANDIDATE_LIMIT else None
    candidate_cutoff = max(clock, costs[anchor]) if anchor is not None else fifo_clock
    mandatory = sorted(
        (i for i in previous if round_number - previous[i]['ready_since_round'] >= MAX_DEFERRED_ROUNDS),
        key=lambda i: (previous[i]['ready_since_round'], previous[i]['ready_since_time'], *order_key(i)),
    )
    required = set(mandatory) if policy == 'weak' else set()

    def phi(ids):
        counts = [0] * len(weights)
        for i in ids:
            counts[groups[i]] += 1
        return math.fsum((n / BUFFER_SIZE - p) ** 2 for n, p in zip(counts, weights))

    def score(ids):
        return (phi(ids), sum(round_number - 1 - stamps[i] for i in ids),
                max(clock, costs[ids[-1]]) - clock, tuple(order_key(i) for i in ids))

    selected = list(baseline)
    mode = 'fifo'
    fallback_reason = None
    if policy == 'weak' and len(mandatory) > BUFFER_SIZE:
        fallback_reason = 'mandatory_overflow'
    elif balance_round:
        if anchor is None:
            fallback_reason = 'insufficient_candidates'
        elif not required.issubset(candidates):
            fallback_reason = 'mandatory_outside_candidates'
        else:
            best = None
            for count in range(1, MAX_REPLACEMENTS + 1):
                for keep in combinations(baseline, BUFFER_SIZE - count):
                    for extra in combinations(candidates[BUFFER_SIZE:], count):
                        ids = sorted(keep + extra, key=order_key)
                        if anchor not in ids or not required.issubset(ids):
                            continue
                        key = score(ids)
                        if best is None or key < best[0]:
                            best = (key, ids)
            if best is None:
                fallback_reason = 'no_feasible_anchor_mandatory_buffer'
            else:
                selected = best[1]
                mode = 'balanced'
    elif policy == 'weak' and not required.issubset(baseline_set):
        fallback_reason = 'mandatory_outside_fifo'
    if fallback_reason is not None:
        if required and (not required.issubset(baseline_set) or len(mandatory) > BUFFER_SIZE):
            chosen = mandatory[:BUFFER_SIZE]
            chosen_set = set(chosen)
            chosen += [i for i in active if i not in chosen_set][:BUFFER_SIZE - len(chosen)]
            selected = sorted(chosen, key=order_key)
            mode = 'mandatory_fallback'
        else:
            mode = 'fifo_fallback'
    selected_set = set(selected)
    decision_clock = max(clock, costs[selected[-1]])
    ready_ids = [i for i in active if costs[i] <= decision_clock]
    observed = {
        i: dict(previous[i]) if i in previous else dict(
            completion_time=costs[i], stamp=stamps[i],
            ready_since_round=round_number, ready_since_time=costs[i])
        for i in ready_ids
    }
    queue_wait = [dict(
        client_id=i, completion_time=costs[i], stamp=stamps[i],
        staleness=round_number - 1 - stamps[i],
        ready_since_round=observed[i]['ready_since_round'],
        ready_since_time=observed[i]['ready_since_time'],
        waited_ready_rounds=round_number - observed[i]['ready_since_round'],
        queue_wait_time=decision_clock - observed[i]['ready_since_time'],
        mandatory=i in required, selected=i in selected_set,
    ) for i in ready_ids]
    added = [i for i in selected if i not in baseline_set]
    removed = [i for i in baseline if i not in selected_set]
    cap_reasons = []
    if policy == 'weak' and len(mandatory) > BUFFER_SIZE:
        cap_reasons.append('mandatory_overflow')
    if len(added) > MAX_REPLACEMENTS:
        cap_reasons.append('replacement_cap')
    record = dict(
        round_number=round_number, policy=policy, mode=mode, phase='exploratory',
        group_reference='offline_population', completion_information='scheduled_completion_times',
        policy_parameters=dict(buffer_size=BUFFER_SIZE, candidate_limit=CANDIDATE_LIMIT,
                               balance_every=BALANCE_EVERY, max_replacements=MAX_REPLACEMENTS,
                               max_deferred_rounds=MAX_DEFERRED_ROUNDS),
        balance_round=balance_round, fallback_reason=fallback_reason,
        selected_ids=selected, selected_real_costs=[costs[i] for i in selected],
        fifo_ids=baseline, fifo_counterfactual_phi=phi(baseline), selected_phi=phi(selected),
        fifo_decision_clock=fifo_clock, candidate_ids=candidates,
        candidate_cutoff_time=candidate_cutoff, cutoff_anchor_id=anchor,
        anchor_retained=anchor in selected_set if anchor is not None else False,
        mandatory_ids=mandatory if policy == 'weak' else [],
        unserved_mandatory_ids=[i for i in mandatory if i not in selected_set] if policy == 'weak' else [],
        replacements=len(added), replacement_ids=added, removed_fifo_ids=removed,
        cap_exception=bool(cap_reasons), cap_exception_reasons=cap_reasons,
        prior_clock=clock, decision_clock=decision_clock, clock_increment=decision_clock - clock,
        ready_ids=ready_ids, selected_subset_ready=selected_set.issubset(ready_ids),
        queue_wait=queue_wait, selected_staleness_sum=sum(round_number - 1 - stamps[i] for i in selected),
        selected_waited_ready_rounds_sum=sum(round_number - observed[i]['ready_since_round'] for i in selected),
        selected_queue_wait_time_sum=math.fsum(decision_clock - costs[i] for i in selected),
    )
    new_state = dict(last_round=round_number, clock=decision_clock,
                     ready={i: dict(item) for i, item in observed.items() if i not in selected_set})
    return record, new_state

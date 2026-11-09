import copy
import random
import unittest

from utils.direction_balanced_buffer import select_buffer


class DirectionBalancedBufferTests(unittest.TestCase):
    def setUp(self):
        self.costs = list(range(1, 31))
        self.stamps = [0] * 30
        self.groups = [0] * 10 + [1] * 20
        self.weights = [.5, .5]

    def choose(self, **changes):
        values = dict(costs=self.costs, stamps=self.stamps, group_ids=self.groups,
                      population_weights=self.weights, round_number=5, clock=0., policy='weak')
        values.update(changes)
        return select_buffer(**values)

    def test_fifo_exact_ties_and_ignores_balance_round(self):
        record, state = self.choose(costs=[2.] * 30, policy='fifo')
        self.assertEqual(record['selected_ids'], list(range(10)))
        self.assertEqual(record['decision_clock'], 2.)
        self.assertFalse(record['balance_round'])
        self.assertEqual(len(state['ready']), 20)

    def test_weak_balances_every_fifth_round_retaining_anchor_with_two_swaps(self):
        decision, _ = self.choose()
        self.assertEqual(decision['mode'], 'balanced')
        self.assertEqual(decision['replacements'], 2)
        self.assertIn(14, decision['selected_ids'])
        self.assertEqual(decision['cutoff_anchor_id'], 14)
        self.assertTrue(decision['anchor_retained'])
        self.assertEqual(decision['decision_clock'], 15.)
        self.assertEqual(decision['selected_real_costs'],
                         [self.costs[i] for i in decision['selected_ids']])
        self.assertLess(decision['selected_phi'], decision['fifo_counterfactual_phi'])
        ordinary, _ = self.choose(round_number=4)
        self.assertEqual(ordinary['selected_ids'], list(range(10)))

    def test_same_state_fifo_counterfactual_is_pure_and_does_not_advance_rng(self):
        before = copy.deepcopy((self.costs, self.stamps, self.groups, self.weights))
        rng = random.getstate()
        first = self.choose()
        fifo = self.choose(policy='fifo')
        second = self.choose()
        self.assertEqual(first, second)
        self.assertNotEqual(first[0]['selected_ids'], fifo[0]['selected_ids'])
        self.assertEqual((self.costs, self.stamps, self.groups, self.weights), before)
        self.assertEqual(random.getstate(), rng)

    def test_ready_item_age_forces_mandatory_on_ordinary_round(self):
        previous = dict(last_round=6, clock=30., ready={
            29: dict(completion_time=30., stamp=0, ready_since_round=5, ready_since_time=30.)})
        decision, state = self.choose(round_number=7, clock=30., state=previous)
        self.assertEqual(decision['mandatory_ids'], [29])
        self.assertIn(29, decision['selected_ids'])
        self.assertEqual(decision['fallback_reason'], 'mandatory_outside_fifo')
        self.assertEqual(decision['mode'], 'mandatory_fallback')
        self.assertNotIn(29, state['ready'])
        item = next(item for item in decision['queue_wait'] if item['client_id'] == 29)
        self.assertEqual(item['waited_ready_rounds'], 2)
        self.assertEqual(item['queue_wait_time'], 0.)

    def test_changed_update_identity_clears_ready_age(self):
        previous = dict(last_round=6, clock=30., ready={
            29: dict(completion_time=30., stamp=0, ready_since_round=5, ready_since_time=30.)})
        costs = list(self.costs)
        costs[29] = 31.
        stamps = list(self.stamps)
        stamps[29] = 6
        decision, _ = self.choose(costs=costs, stamps=stamps, round_number=7, clock=30., state=previous)
        self.assertEqual(decision['mandatory_ids'], [])
        self.assertNotIn(29, decision['selected_ids'])

    def test_mandatory_overflow_records_cap_exception(self):
        ready = {i: dict(completion_time=float(self.costs[i]), stamp=0,
                         ready_since_round=1, ready_since_time=float(self.costs[i]))
                 for i in range(19, 30)}
        decision, _ = self.choose(round_number=4, clock=30.,
                                  state=dict(last_round=3, clock=30., ready=ready))
        self.assertEqual(decision['selected_ids'], list(range(19, 29)))
        self.assertEqual(decision['unserved_mandatory_ids'], [29])
        self.assertEqual(decision['fallback_reason'], 'mandatory_overflow')
        self.assertTrue(decision['cap_exception'])
        self.assertIn('replacement_cap', decision['cap_exception_reasons'])

    def test_insufficient_candidates_uses_explicit_fifo_fallback(self):
        decision, _ = self.choose(costs=self.costs[:12], stamps=self.stamps[:12],
                                  group_ids=self.groups[:12])
        self.assertEqual(decision['selected_ids'], list(range(10)))
        self.assertEqual(decision['fallback_reason'], 'insufficient_candidates')

    def test_invalid_inputs_and_noncausal_state_are_rejected(self):
        invalid = [dict(costs=[float('nan')] * 30), dict(costs=[-1.] * 30),
                   dict(stamps=[5] * 30), dict(population_weights=[.1, .1]),
                   dict(group_ids=[2] * 30), dict(costs=set(range(1, 31))),
                   dict(round_number=True), dict(clock=-1.),
                   dict(state=dict(last_round=1, clock=0., ready={})),
                   dict(round_number=3, clock=2., state=dict(last_round=2, clock=3., ready={})),
                   dict(round_number=3, clock=30., state=dict(last_round=2, clock=30., ready={
                       0: dict(completion_time=1., stamp=1, ready_since_round=1, ready_since_time=1.)}))]
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.choose(**change)


if __name__ == '__main__':
    unittest.main()

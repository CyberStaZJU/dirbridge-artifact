# FedScale device profile

Place the FedScale client device-capacity payload here as:

```text
fedscale_device_info/client_device_capacity
```

The full original FedScale profile can be large, so it is not required for a minimal repository. To make the exact mapping explicit, use:

```bash
python scripts/export_client_selection.py \
  --dataset femnist \
  --data-root data/femnist_pt \
  --num-clients 1000 \
  --min-samples-per-client 250 \
  --fedscale-profile fedscale_device_info/client_device_capacity \
  --seed 1 \
  --out-dir data/client_selection
```

and similarly for GSpeech. This exports the selected dataset clients, the local-client-index to FedScale-profile-client-id mapping, and a compact profile subset for the selected FedScale profiles.

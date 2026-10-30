from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb
import json, collections
ds = datastore.DataStore()
ds.open_for_scan("/mnt/c/YangJaehong/rotating-without-seeing/wandb/run-20261028_152255-lxgr93nn/run-lxgr93nn.wandb")
kinds = collections.Counter(); last_gs = None; n_hist = 0; err = None
while True:
    try:
        data = ds.scan_data()
    except Exception as e:
        err = repr(e); break
    if data is None: break
    rec = pb.Record(); rec.ParseFromString(data)
    k = rec.WhichOneof("record_type"); kinds[k] += 1
    if k == "history":
        n_hist += 1
        for it in rec.history.item:
            if it.key == "global_step" or it.nested_key == ["global_step"]:
                last_gs = it.value_json
print(kinds.most_common(12)); print("history records:", n_hist, "last global_step:", last_gs, "scan error:", err)

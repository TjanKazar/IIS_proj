import sys
import great_expectations as gx

context = gx.get_context()

datasource_name = "traffic"
data_asset_name = "traffic_data"

#load the data asset
asset = context.get_datasource(datasource_name).get_asset(data_asset_name)

# load checkpoint
checkpoint_name = "traffic_checkpoint"
checkpoint = context.get_checkpoint(checkpoint_name)

# run checkpoint
run_id = "traffic_checkpoint_run"
checkpoint_result = checkpoint.run(
    run_id=run_id
)

#build data docs
context.build_data_docs()

#check if checkpoint passed
if checkpoint_result["success"]:
    print("Checkpoint passed!")
    sys.exit(0)
else:
    print("Checkpoint failed!")
    sys.exit(1)

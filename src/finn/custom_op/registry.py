# make sure new CustomOp subclasses are imported here so that they get
# registered and plug in correctly into the infrastructure
from finn.custom_op.multithreshold import MultiThreshold

# create a mapping of all known CustomOp names and classes
custom_op = {}

custom_op["MultiThreshold"] = MultiThreshold

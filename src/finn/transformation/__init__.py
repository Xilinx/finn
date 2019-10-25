"""
Guide to writing FINN transformations
-------------------------------------

* Your transformation should take in an ONNX model, and return a tuple with
 (transformed_model: ModelProto, model_was_changed: Bool)
* The original model should not be modified, use e.g. copy.deepcopy() if you
  want to work on a copy of the graph for modifications.
* model_was_changed indicates whether your transformation made any changes to
  the model. If you know your transformation needs to be called only once and
  repeated calls have no further effect, you can return False even if the model
  was changed.
* You MUST return model_was_changed=False at some point when your transformation
  is called multiple times, otherwise apply_repeated() will loop infinitely.
* If you cannot guarantee that the transformation will reach a fixed point,
  you must declare this and return only the transformed model instead of a tuple.
"""

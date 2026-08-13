import numpy as np

class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            x = request.get_input_tensor("INPUT").as_numpy()
            y = np.sum(x, axis=1, keepdims=True)
            responses.append(
                request.get_output_tensor("OUTPUT", y)
            )
        return responses
import unittest

from project_paths import (
    DEFAULT_ONNX_NAME,
    DEFAULT_WEIGHT_NAME,
    DEPLOY_MODELS_DIR,
    MODELS_DIR,
    candidate_weight_paths,
    resolve_onnx_path,
    resolve_weights_path,
)


class ProjectPathsTest(unittest.TestCase):
    def test_candidate_weight_paths_order(self):
        self.assertEqual(
            candidate_weight_paths(),
            [
                MODELS_DIR / DEFAULT_WEIGHT_NAME,
                DEPLOY_MODELS_DIR / DEFAULT_WEIGHT_NAME,
            ],
        )

    def test_default_weight_resolution_prefers_models_directory(self):
        self.assertEqual(
            resolve_weights_path(),
            MODELS_DIR / DEFAULT_WEIGHT_NAME,
        )

    def test_default_onnx_resolution_points_to_deploy_models(self):
        self.assertEqual(
            resolve_onnx_path(),
            DEPLOY_MODELS_DIR / DEFAULT_ONNX_NAME,
        )


if __name__ == "__main__":
    unittest.main()

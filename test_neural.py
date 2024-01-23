import unittest
from unittest.mock import patch, MagicMock
import torch
import numpy as np
import pandas as pd
from training.training import IrisClassifier, prepare_data, create_data_loaders, evaluate_model, train_batch
from inference.inference import IrisClassifier, load_model, softmax, run_inference
from data_process.data_processing import generate_iris_data

class TestModelEvaluation(unittest.TestCase):

    def test_evaluate_model(self):
        model = IrisClassifier()
        X_test = torch.rand(5, 4)
        y_test = torch.randint(0, 3, (5,))
        accuracy = evaluate_model(model, X_test, y_test)
        self.assertIsInstance(accuracy, float)

class TestTraining(unittest.TestCase):

    def test_train_batch(self):
        model = IrisClassifier()
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()
        X_batch = torch.rand(5, 4)
        y_batch = torch.randint(0, 3, (5,))
        train_batch(model, X_batch, y_batch, optimizer, criterion)

class TestLoadModel(unittest.TestCase):

    @patch('os.path.exists', return_value=False)
    def test_load_model_file_not_found_error(self, _):
        with self.assertRaises(FileNotFoundError):
            load_model('fake_path')

class TestSoftmaxFunction(unittest.TestCase):
    def test_softmax_output(self):
        logits = np.array([[10, 2, 8]])
        probabilities = softmax(logits)
        np.testing.assert_almost_equal(np.sum(probabilities), 1.0)

class TestRunInference(unittest.TestCase):
    def test_run_inference_output(self):
        model = IrisClassifier()
        model.eval()  
        input_tensor = torch.rand(1, 4)  
        with torch.no_grad():  
            output = run_inference(model, input_tensor)
        self.assertEqual(output.shape, (1, 3))  

class TestGenerateIrisData(unittest.TestCase):

    @patch('pandas.read_html')
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    def test_generate_iris_data_success(self, mock_to_csv, mock_makedirs, mock_exists, mock_read_html):
        mock_exists.return_value = True
        mock_read_html.return_value = [pd.DataFrame({
            'Sepal length': [5.1, 4.9],
            'Sepal width': [3.5, 3.0],
            'Petal length': [1.4, 1.4],
            'Petal width': [0.2, 0.2],
            'Species': ['Iris-setosa', 'Iris-versicolor']
        })]

        settings = {
            'general': {'data_dir': '/fakepath', 'random_state': 1},
            'train': {'test_size': 0.2, 'table_name': 'train.csv'},
            'inference': {'inp_table_name': 'inference.csv'}
        }
        generate_iris_data(settings)


        mock_exists.assert_called_with('/fakepath')

        calls = [unittest.mock.call(f'/fakepath/train.csv', index=False),
                 unittest.mock.call(f'/fakepath/inference.csv', index=False)]
        mock_to_csv.assert_has_calls(calls, any_order=True)

    @patch('pandas.read_html')
    def test_generate_iris_data_exception_handling(self, mock_read_html):
        mock_read_html.side_effect = Exception('Test exception')

        settings = {
            'general': {'data_dir': '/fakepath', 'random_state': 1},
            'train': {'test_size': 0.2, 'table_name': 'train.csv'},
            'inference': {'inp_table_name': 'inference.csv'}
        }

        with self.assertLogs(level='ERROR') as log:
            generate_iris_data(settings)
            self.assertIn('An error occurred: Test exception', log.output[0])


if __name__ == '__main__':
    unittest.main()

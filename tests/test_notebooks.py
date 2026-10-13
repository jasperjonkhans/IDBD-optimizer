import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src import main
from src.data import DataStream


class MomentumNotebookTests(unittest.TestCase):
    def test_notebook_uses_shared_api_and_retains_sampled_traces(self):
        path = Path(main.__file__).parents[1] / 'notebooks' / 'idbd_momentum.ipynb'
        notebook = json.loads(path.read_text())
        namespace = {}
        # Execute real notebook sources, except IPython magics, with a small budget.
        for index in (1, 2, 3):
            source = ''.join(notebook['cells'][index]['source'])
            source = '\n'.join(line for line in source.splitlines() if not line.startswith('%'))
            exec(compile(source, f'{path}:cell-{index}', 'exec'), namespace)
        namespace.update(SEED=1234, INPUT_DIM=4, ITERATIONS=7, BATCH_SIZE=3,
                         RECORD_EVERY=3, TEST_ITERATIONS=2, TEST_BATCH_SIZE=5,
                         RUN_MATCHED_META_LR=True)
        rng_before = torch.random.get_rng_state().clone()
        namespace.update(train=Mock(wraps=main.train), evaluate=Mock(wraps=main.evaluate))
        with patch.object(main.mlflow, 'log_metrics') as log, \
                patch.object(plt, 'show'), contextlib.redirect_stdout(io.StringIO()):
            for index in (4, 6, 8, 9, 11):
                exec(''.join(notebook['cells'][index]['source']), namespace)
            self.assertEqual(namespace['train'].call_count, 2)
            self.assertEqual(namespace['evaluate'].call_count, 2)
            self.assertFalse(namespace['train'].call_args.kwargs['log_mlflow'])
            self.assertFalse(namespace['train'].call_args.kwargs['record_history'])
        plt.close('all')
        log.assert_not_called()
        torch.testing.assert_close(torch.random.get_rng_state(), rng_before)
        result = namespace['results']
        np.testing.assert_array_equal(result['steps'], [0, 3, 6, 7])
        np.testing.assert_array_equal(result['examples'], [0, 9, 18, 21])
        self.assertEqual(result['train_metrics']['examples'].tolist(), [21] * 4)
        self.assertEqual(result['metrics']['examples'].tolist(), [10] * 4)
        self.assertEqual(len(result['models']), 3)
        for name, model in result['models'].items():
            trace = result['trace'][name]
            self.assertEqual(trace['weights'].shape, (4, 4))
            self.assertEqual(trace['rates'].shape, (4, 4))
            self.assertTrue(np.isnan(trace['loss_ma_200'][0]))
            self.assertTrue(np.isfinite(trace['loss_ma_200'][1:]).all())
            np.testing.assert_allclose(trace['weights'][-1], model.weight.detach().flatten())
            np.testing.assert_allclose(trace['rates'][-1],
                                       main.learning_rates(model, result['optimizers'][name]))
        expected = main.evaluate(result['models'], DataStream(4, seed=1235), 2, 5)
        for column in ('clean_mse', 'noisy_mse', 'active_clean_mse'):
            pd.testing.assert_series_equal(result['metrics'][f'test_{column}'],
                                           expected[column], check_names=False)
        matched = namespace['matched_results']
        for name in ('SGD', 'IDBD'):
            np.testing.assert_array_equal(result['trace'][name]['weights'],
                                          matched['trace'][name]['weights'])


if __name__ == '__main__':
    unittest.main()

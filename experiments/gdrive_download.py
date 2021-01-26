from sacred import Experiment
import os

from fs.data.utils import load_gdrive_file

ex = Experiment()


@ex.main
def saved_model(testing_dataset, model_id, _run, _log, batching=False, validation=False):
    load_gdrive_file(model_id, 'zip')

if __name__ == '__main__':
    ex.run_commandline()
    os._exit(os.EX_OK)

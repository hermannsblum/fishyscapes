from sacred.observers import MongoObserver, FileStorageObserver
import fs.settings as settings
import os
import shutil
from logging import getLogger, info
from contextlib import contextmanager

from .utils import ExperimentData


def load_data(**data_config):
    """
    Load the data specified in the data_config dict.
    """
    name = data_config.pop('name', 'fsdata')
    subset = data_config.pop('subset', 'testset')
    data = get_dataset(name)(**data_config)
    if subset == 'measureset':
        info('Using measureset')
        ds = data.get_measureset()
    elif subset == 'validation_set':
        info('Using validation set')
        ds = data.get_validation_set()
    elif subset == 'trainset':
        info('Using training set')
        ds = data.get_trainset()
    elif subset == 'testset':
        info('Using testset')
        ds = data.get_testset()
    else:
        raise UserWarning('Data subset {} is no correct identifier'.format(subset))
    return data.get_data_description(), ds


def get_observer():
    if hasattr(settings, 'EXPERIMENT_DB_HOST') and settings.EXPERIMENT_DB_HOST:
        print('mongo observer created', flush=True)
        return MongoObserver.create(url='mongodb://{user}:{pwd}@{host}/{db}'.format(
                                        host=settings.EXPERIMENT_DB_HOST,
                                        user=settings.EXPERIMENT_DB_USER,
                                        pwd=settings.EXPERIMENT_DB_PWD,
                                        db=settings.EXPERIMENT_DB_NAME),
                                    db_name=settings.EXPERIMENT_DB_NAME)
    elif hasattr(settings, 'EXPERIMENT_STORAGE_FOLDER') \
            and settings.EXPERIMENT_STORAGE_FOLDER:
        return FileStorageObserver.create(settings.EXPERIMENT_STORAGE_FOLDER)
    else:
        raise UserWarning("No observer settings found.")


def import_weights_into_network(net, starting_weights, prefix=False):
    """Based on either a list of descriptions of training experiments or one description,
    load the weights produced by these trainigns into the given network.

    Args:
        net: An instance of a `base_model` inheriting class.
        starting_weights: Either dict or list of dicts.
            if dict: expect key 'experiment_id' to match a previous experiment's ID.
                if key 'filename' is not set, will search for the first artifact that
                has 'weights' in the name.
            if list: a list of dicts where each dict will be evaluated as above
        kwargs are passed to net.import_weights
    """
    log = getLogger('weight import')

    if isinstance(starting_weights, list):
        for weights in starting_weights:
            import_weights_into_network(net, weights)
    elif isinstance(starting_weights, dict):
        for prefix, weights in starting_weights.items():
            import_weights_into_network(net, weights, prefix=prefix)
    elif isinstance(starting_weights, str):
        log.info('Importing network weights %s' % starting_weights)
        if starting_weights == 'paul_adapnet':
            net.import_weights(os.path.join(settings.DATA_BASEPATH,
                                            'Adapnet_weights_160000.npz'),
                               chill_mode=True, translate_prefix=prefix)
        elif starting_weights == 'imagenet_adapnet':
            net.import_weights(os.path.join(settings.DATA_BASEPATH,
                                            'resnet50_imagenet.npz'),
                               chill_mode=True, translate_prefix=prefix)
        elif starting_weights == 'deeplab_cityscapes':
            net.import_weights(os.path.join(settings.DATA_BASEPATH,
                                            'deeplab_models/cityscapes_trainfine.npz'),
                               chill_mode=True, translate_prefix=prefix)
        elif os.path.isdir(starting_weights) or starting_weights.endswith('.zip'):
            # try to load string as experiment
            log.info('Importing weights from experiment %s' % starting_weights)
            training_experiment = ExperimentData(starting_weights)
            net.import_weights(training_experiment.get_weights(), translate_prefix=prefix)
        else:
            net.import_weights(os.path.join(settings.DATA_BASEPATH, starting_weights),
                               chill_mode=True, add_prefix=prefix)
    else:  # description is an experiment id
        log.info('Importing weights from experiment %s' % starting_weights)
        training_experiment = ExperimentData(starting_weights)
        net.import_weights(training_experiment.get_weights(), translate_prefix=prefix)


def create_directories(run):
    """
    Make sure directories for storing diagnostics are created and clean.

    Args:
        run_id: ID of the current sacred run, you can get it from _run._id in a captured
            function.
        experiment: The sacred experiment object
    Returns:
        The path to the created output directory you can store your diagnostics to.
    """
    root = settings.EXP_OUT
    # create temporary directory for output files
    if not os.path.exists(root):
        os.makedirs(root)
    # The id of this experiment is stored in the magical _run object we get from the
    # decorator.
    output_dir = '{}/{}'.format(root, run._id)
    if run._id is None and os.path.exists(output_dir):
        # Directory may already exist if run_id is None (in case of an unobserved
        # test-run)
        shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Tell the experiment that this output dir is also used for tensorflow summaries
    run.info.setdefault("tensorflow", {}).setdefault("logdirs", [])\
        .append(output_dir)
    return output_dir


def clear_directory(run):
    output_dir = '{}/{}'.format(settings.EXP_OUT, run._id)
    if not os.path.exists(output_dir):
        # nothing to do
        return
    shutil.rmtree(output_dir)


@contextmanager
def experiment_context(modelname, net_config, data_description, _run, _seed,
                       starting_weights=None):
    output_dir = create_directories(_run)
    model = get_model(modelname)
    _run.info['tf_random_seed'] = _seed
    try:
        with model(data_description=data_description,
                   output_dir=output_dir,
                   random_seed=_seed,
                   **net_config) as net:
            if starting_weights is not None:
                import_weights_into_network(net, starting_weights)
            yield net
    finally:
        # To end the experiment, we collect all produced output files and store them.
        for filename in os.listdir(output_dir):
            _run.add_artifact(os.path.join(output_dir, filename))


def setup_experiment(experiment, main=False):
    def _setup_experiment(f):
        def experiment_setup(modelname, net_config, _run, _seed, dataset=None,
                             starting_weights=None):
            output_dir = create_directories(_run)

            if dataset is not None:
                data = get_dataset(dataset['name'])
            model = get_model(modelname)
            _run.info['tf_random_seed'] = _seed
            with model(data_description=data.get_data_description(),
                       output_dir=output_dir,
                       random_seed=_seed,
                       **net_config) as net:
                if starting_weights is not None:
                    import_weights_into_network(net, starting_weights)
                if dataset is not None:
                    f(net, data)
                else:
                    f(net)
            # To end the experiment, we collect all produced output files and store them.
            for filename in os.listdir(output_dir):
                _run.add_artifact(os.path.join(output_dir, filename))

        function_for_sacred = experiment_setup
        function_for_sacred.__name__ = f.__name__
        if main:
            function_for_sacred = experiment.main(function_for_sacred)
        else:
            function_for_sacred = experiment.command(function_for_sacred)
        return function_for_sacred
    return _setup_experiment

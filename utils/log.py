import os
import logging

def setup_logging(log_file='log.txt', resume=False, dummy=False):
    """
    Setup logging configuration
    """
    if dummy:
        logging.getLogger('dummy')
    else:
        if os.path.isfile(log_file) and resume:
            file_mode = 'a'
        else:
            file_mode = 'w'

        root_logger = logging.getLogger()
        if root_logger.handlers:
            root_logger.removeHandler(root_logger.handlers[0])
        logging.basicConfig(level=logging.DEBUG,
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S",
                            filename=log_file,
                            filemode=file_mode)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

class ResultsLog(object):

    supported_data_formats = ['csv', 'json']

    def __init__(self, path='', title='', params=None, resume=False, data_format='csv'):
        """
        Parameters
        ----------
        path: string
            path to directory to save data files
        plot_path: string
            path to directory to save plot files
        title: string
            title of HTML file
        params: Namespace
            optionally save parameters for results
        resume: bool
            resume previous logging
        data_format: str('csv'|'json')
            which file format to use to save the data
        """
        if data_format not in ResultsLog.supported_data_formats:
            raise ValueError('data_format must of the following: ' +
                             '|'.join(['{}'.format(k) for k in ResultsLog.supported_data_formats]))

        if data_format == 'json':
            self.data_path = '{}.json'.format(path)
        else:
            self.data_path = '{}.csv'.format(path)
        if params is not None:
            export_args_namespace(params, '{}.json'.format(path))
        self.plot_path = '{}.html'.format(path)
        self.results = None
        self.clear()
        self.first_save = True
        if os.path.isfile(self.data_path):
            if resume:
                self.load(self.data_path)
                self.first_save = False
            else:
                os.remove(self.data_path)
                self.results = pd.DataFrame()
        else:
            self.results = pd.DataFrame()

        self.title = title
        self.data_format = data_format

        if HYPERDASH_AVAILABLE:
            name = self.title if title != '' else path
            self.hd_experiment = hyperdash.Experiment(name)
            if params is not None:
                for k, v in params._get_kwargs():
                    self.hd_experiment.param(k, v, log=False)
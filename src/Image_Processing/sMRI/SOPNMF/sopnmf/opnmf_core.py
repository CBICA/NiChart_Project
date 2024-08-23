from sopnmf.workflow import VB_OPNMF, VB_OPNMF_mini_batch

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2023"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.0.4"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

def opnmf(participant_tsv, output_dir, tissue_binary_mask, num_component_min, num_component_max, num_component_step=1, max_iter=50000,
          init_method='NNDSVDRandomSVD', magnitude_tolerance=0, early_stopping_epoch=20, n_threads=8, verbose=False):
    """
    Core function of opNMF algorithm.
    :param participant_tsv: str, path to the tsv containing the population information, insipred by BIDS convention. The tsv contains
        the following headers: "
                                 "i) the first column is the participant_id;"
                                 "ii) the second column should be the session_id;"
                                 "iii) the third column should be the path, image path for each subject"
    :param output_dir: str, path to the output folder
    :param tissue_binary_mask: str. This is a tissue binary mask that constrains the voxels only in desired regions. During training,
            the mask was made based on all images by taking off all 0 voxels, which is quite loose. In the new code, we force to use
            a tissue binary mask created either from the template image or from the population based mean image.
    :param num_component_min: int, the minimum number of components
    :param num_component_max: int, the maximum number of components
    :param num_component_step: int, default is 1
    :param max_iter: int, maximum number of iterations for convergence
    :param init_method: str, default is NNDSVDRandomSVD ['NNDSVDRandomSVD', 'Random']
    :param magnitude_tolerance: float, the tolerance of loss change magnitude
    :param early_stopping_epoch: int, the tolerance of number of bad epochs for early stopping
    :param n_threads: int, default is 8. The number of threads to run model in parallel.
    :param verbose: Bool, default is False. If the output message is verbose.
    :return:
    """

    ### For voxel approach
    print('Performing OPNMF for voxel-wise features...')
    # ## Here, semi-supervised clustering
    wf = VB_OPNMF(tissue_binary_mask, output_dir, participant_tsv, num_component_min, num_component_max, num_component_step, init_method,
                  max_iter, magnitude_tolerance, early_stopping_epoch, n_threads, verbose)

    wf.run()

    print('Finish...')

def sopnmf(participant_tsv, participant_tsv_max_memory, output_dir, tissue_binary_mask, num_component_min, num_component_max,
                     num_component_step=1, batch_size=8, max_epoch=50000, init_method='NNDSVDRandomSVD', magnitude_tolerance=0,
                     early_stopping_epoch=100, n_threads=8, verbose=False):
    """
    Core function of sopNMF algorithm. Train the model like a deep learning model. Normally, with smaller batch size, it converges faster,
    since reading the batch data by CPU is less time-consuming (even though your models will have to run more iterations).
    :param participant_tsv: str, path to the tsv containing the population information, insipred by BIDS convention. The tsv contains
        the following headers: "
                                 "i) the first column is the participant_id;"
                                 "ii) the second column should be the session_id;"
                                 "iii) the third column should be the path, image path for each subject"
    :param participant_tsv_max_memory: str, path to the tsv containing the maximum number of sub-population for the create of the mask and initialization of W matrix
    :param output_dir: str, path to the output folder
    :param tissue_binary_mask: str. This is a tissue binary mask that contrains the voxels only in desired regions. During training,
        the mask was made based on all images by taking off all 0 voxels, which is quite loose. In the new code, we force to use
        a tissue binary mask created either from the template image or from the population based mean image.
    :param num_component_min: int, the minimum number of components
    :param num_component_max: int, the maximum number of components
    :param num_component_step: int, default is 1
    :param batch_size: int, batch size for the stochastic Lagrangian update rules.
    :param max_epoch: int, maximum number of epochs for convergence
    :param init_method: str, default is NNDSVDRandomSVD ['NNDSVDRandomSVD', 'Random']
    :param early_stopping_epoch: the tolerance for validation loss not decreasing.
    :param n_threads: int, default is 8. The number of threads to run model in parallel.
    :param verbose: Bool, default is False. If the output message is verbose.
    :return:
    """

    ### For voxel approach
    print('Performing OPNMF for voxel-wise features...')
    # ## Here, semi-supervised clustering
    wf = VB_OPNMF_mini_batch(tissue_binary_mask, output_dir, participant_tsv, participant_tsv_max_memory, num_component_min, num_component_max,
                             num_component_step, batch_size, init_method, max_epoch, magnitude_tolerance, early_stopping_epoch, n_threads, verbose)

    wf.run()

    print('Finish...')
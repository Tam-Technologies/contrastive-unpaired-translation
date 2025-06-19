import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from google.cloud import aiplatform

import constants

TENSORBOARD_NAME = 'CUTTensorboard'
TENSORBOARD_RESOURCE_NAME = 'projects/437403722141/locations/us-central1/tensorboards/2914176404583088128'

def create_tensorboard():
    tensorboard = aiplatform.Tensorboard.create(
        display_name=TENSORBOARD_NAME, project=constants.PROJECT_ID, location=constants.LOCATION
    )
    return tensorboard.gca_resource.name

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Run Vertex AI training job')
    parser.add_argument("image", type=str, help='Docker image SHA')
    parser.add_argument("--experiment_name", type=str, required=True, help='Name of experiment')
    parser.add_argument("--dataset_name", type=str, required=True, help='Name of dataset to train on')
    parser.add_argument("--n_epochs", default=25, type=int, help='Number of training epochs to run')
    args = parser.parse_args()

    if not TENSORBOARD_RESOURCE_NAME:
        TENSORBOARD_RESOURCE_NAME = create_tensorboard() # Set TENSORBOARD_RESOURCE_NAME above in order to reuse the same tensorboard

    aiplatform.init(project=constants.PROJECT_ID, location=constants.LOCATION,
                    staging_bucket=constants.VERTEX_AI_BUCKET_NAME)

    job = aiplatform.CustomContainerTrainingJob(
        display_name='CUTTraining',
        container_uri=f'gcr.io/{constants.PROJECT_ID}/{constants.IMAGE_NAME}@sha256:{args.image}',
        location=constants.LOCATION
    )

    job_args = ['train',
                '--dataroot', f'/gcs/{constants.VERTEX_AI_BUCKET_NAME}',
                '--dataroot_B', f'/gcs',
                '--dataset_mode', 'silhouette',
                '--dataset_csv_A',
                f'/gcs/{constants.VERTEX_AI_BUCKET_NAME}/cyclegan_dataset_csv/{args.dataset_name}/synthetic_silhouettes.csv',
                '--dataset_csv_B',
                f'/gcs/{constants.VERTEX_AI_BUCKET_NAME}/cyclegan_dataset_csv/{args.dataset_name}/real_silhouettes.csv',
                '--checkpoints_dir',
                f'/gcs/{constants.VERTEX_AI_BUCKET_NAME}/cut_checkpoints',
                '--name',
                args.experiment_name,
                '--save_latest_freq', '4000',
                '--save_epoch_freq', '1',
                '--display_freq', '400',
                '--update_html_freq', '1000',
                '--print_freq', '100',
                '--display_id', '-1',
                '--n_epochs', f'{args.n_epochs}',
                '--n_epochs_decay', f'{args.n_epochs}',
                '--batch_size', '12',
                '--num_threads', '4',
                '--gpu_ids', '0,1,2,3',
                '--output_nc', '1',
                '--input_nc', '1']

    model = job.run(args=job_args,
                    replica_count=1,
                    machine_type='n1-standard-8',
                    accelerator_type='NVIDIA_TESLA_T4',
                    accelerator_count=4,
                    service_account=constants.VERTEX_AI_SERVICE_ACCOUNT_EMAIL,
                    tensorboard=TENSORBOARD_RESOURCE_NAME)

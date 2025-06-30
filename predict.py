import os
import multiprocessing


import firebase_admin
import firebase_admin.firestore
import numpy as np
from google.cloud import storage
from PIL import Image

import constants
from data import create_dataset
from data.silhouette_dataset import normalize_silhouette_image
from models import create_model
from options.test_options import TestOptions
from util import util

from torchvision.transforms import ToTensor

# Use the application default credentials to initialize app
cred = firebase_admin.credentials.ApplicationDefault()
try:
    firebase_admin.initialize_app(cred, {'projectId': constants.PROJECT_ID})
except:
    pass

db = firebase_admin.firestore.client()
storage_client = storage.Client()
vertex_ai_bucket = storage_client.bucket(constants.VERTEX_AI_BUCKET_NAME)
def overlay_silhouette_images(image1, image2):
    assert image1.size == image2.size, "Images must be the same size"
    size = image1.size

    # Create first silhouette in blue
    first_silhouette = Image.new('RGB', size, (255, 255, 255))
    blue = Image.new('RGB', size, (0, 0, 255))
    first_silhouette.paste(blue, (0, 0), mask=image1)

    # Create second silhouette in yellow
    second_silhouette = Image.new('RGB', size, (255, 255, 255))
    yellow = Image.new('RGB', size, (255, 255, 0))
    second_silhouette.paste(yellow, (0, 0), mask=image2)

    composite = Image.blend(first_silhouette, second_silhouette, alpha=0.5)
    return composite


def transfer_synthetic_silhouette(image_path, model):
    input_A = Image.open(image_path).convert('L')
    real_A_tensor = normalize_silhouette_image(input_A, out_image_size=input_A.size)
    real_A = Image.fromarray(real_A_tensor.squeeze(0).squeeze(0).numpy() * 255).convert('L')
    prediction_A_tensor = normalize_silhouette_image(input_A).unsqueeze(0)
    data = {'A': prediction_A_tensor, 'A_paths': image_path}
    model.set_input(data)  # unpack data from data loader
    model.test()  # run inference
    visuals = model.get_current_visuals()
    im_data = visuals['fake_B']
    im = Image.fromarray(im_data.squeeze(0).squeeze(0).numpy() * 255).convert('L')

    # Center image and pad to original silhouette image size
    square_dim = max(input_A.size[0], input_A.size[1])
    resized_image = im.resize((square_dim, square_dim))
    fake_B = Image.new('L', (input_A.size[0], input_A.size[1]))
    fake_B.paste(resized_image, (int((input_A.size[0] - square_dim) / 2), int((input_A.size[1] - square_dim) / 2)))
    return real_A, fake_B


def process_scan(scanId, model):
    scan_doc = db.collection('synthetic_scans').document(scanId).get()

    scan_data = scan_doc.to_dict()
    if 'cut_version' in scan_data and scan_data['cut_version'] == constants.CUT_VERSION:
        return

    version_string = f"v{constants.CUT_VERSION}"

    updated_scan_data = {'realistic_body_silhouette_paths': [], 'cut_version': constants.CUT_VERSION}
    for image_idx, img_path in enumerate(scan_data['body_silhouette_paths']):
        temp_silhouette_path = f'/tmp/silhouette_{np.random.randint(1e6, 1e7)}.png'
        silhouette_blob = vertex_ai_bucket.get_blob(img_path)
        silhouette_blob.download_to_filename(temp_silhouette_path)
        real_A, fake_B = transfer_synthetic_silhouette(temp_silhouette_path, model)
        os.remove(temp_silhouette_path)
        temp_realistic_silhouette_path = f'/tmp/silhouette_{np.random.randint(1e6, 1e7)}.png'
        fake_B.save(temp_realistic_silhouette_path)
        vertex_ai_bucket.blob(
            f'synthetic_scans/{scanId}/body/realistic_silhouettes_{version_string}/silhouette{image_idx}.png').upload_from_filename(
            temp_realistic_silhouette_path)
        os.remove(temp_realistic_silhouette_path)
        updated_scan_data['realistic_body_silhouette_paths'].append(
            f'synthetic_scans/{scanId}/body/realistic_silhouettes_{version_string}/silhouette{image_idx}.png')

    scan_doc.reference.update(updated_scan_data)

def wrap_try_process_scan(inputs):
    scanId, model = inputs
    try:
        process_scan(scanId, model)
    except Exception as err:
        print(f"Failed to process {scanId}: {err}")

def process_local_scanId(inputs):
    scanId, model, data_dir = inputs
    in_dir = os.path.join(data_dir, f'{scanId}/body/silhouettes')
    out_dir = os.path.join(data_dir, f'{scanId}/body/cut_results')
    train_dir = os.path.join(data_dir, f'{scanId}/body/cut_realistic_silhouettes')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    else:
        print(f"{scanId} has already been processed, skipping")
        return
    for i in range(4):
        A_path = os.path.join(in_dir, f'silhouette{i}.png')
        real_A, fake_B = transfer_synthetic_silhouette(A_path, model)
        comparison_im = overlay_silhouette_images(real_A, fake_B)
        for label, im in zip(['real', 'fake', 'comparison'], [real_A, fake_B, comparison_im]):
            image_name = '%s_%s.png' % (os.path.basename(A_path).split('.')[0], label)
            save_path = os.path.join(out_dir, image_name)
            im.save(save_path)
            if label == 'fake':
                im.save(os.path.join(train_dir, f'silhouette{i}.png'))

if __name__ == "__main__":
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to output files  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    model.parallelize()
    if opt.eval:
        model.eval()

    scanId = 'SCANID_HERE'
    process_local_scanId((scanId, model, opt.dataroot))

    # scanIds = [doc.id for doc in
    #            db.collection('synthetic_scans').where('gender', '==', 'female')
    #            .where('version', '==', '6.2').stream()]
    #
    # print(f"Processing {len(scanIds)} scans")
    # with multiprocessing.Pool() as p:
    #     for _ in tqdm.tqdm(p.imap_unordered(wrap_try_process_scan,
    #                                         ((scanId, model) for scanId in scanIds)),
    #                        total=len(scanIds)):
    #         pass

    print('Done')

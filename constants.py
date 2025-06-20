IMAGE_NAME = 'contrastive-unpaired-translation'

# Google Cloud project settings
LOCATION = 'us-central1'
PROJECT_ID = 'truetoform-dev'
PROJECT_ID_PROD = 'truetoform-6685b'
VERTEX_AI_BUCKET_NAME = 'vertex-ai-data-us-central1'
DEV_SCANS_BUCKET_NAME = f'{PROJECT_ID}.appspot.com'
PROD_SCANS_BUCKET_NAME = f'{PROJECT_ID_PROD}.appspot.com'
VERTEX_AI_SERVICE_ACCOUNT_EMAIL = f'vertex-ai-service-account@{PROJECT_ID}.iam.gserviceaccount.com'

CUT_VERSION = '1'
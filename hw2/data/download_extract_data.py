from google_drive_downloader import GoogleDriveDownloader as gdd


gdd.download_file_from_google_drive(
    file_id='1Fm-avdeNgzhPxhvia0iw9yZzcoOggy7I',
    dest_path='./test.zip', unzip=True)
gdd.download_file_from_google_drive(
    file_id='1lrKueI4HrySQDGvpkilQN9BfaMUN7hZi',
    dest_path='./train.zip', unzip=True)

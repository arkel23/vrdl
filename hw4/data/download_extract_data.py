from google_drive_downloader import GoogleDriveDownloader as gdd


gdd.download_file_from_google_drive(
    file_id='1GL_Rh1N-WjrvF_-YOKOyvq0zrV6TF4hb',
    dest_path='./datasets.zip', unzip=True)

gdd.download_file_from_google_drive(
    file_id='1RtyIeUFTyW8u7oa4z7a0lSzT3T1FwZE9',
    dest_path='./Set5.zip', unzip=True)

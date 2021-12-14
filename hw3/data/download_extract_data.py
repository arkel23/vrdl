from google_drive_downloader import GoogleDriveDownloader as gdd


gdd.download_file_from_google_drive(
    file_id='1nEJ7NTtHcCHNQqUXaoPk55VH3Uwh4QGG',
    dest_path='./dataset.zip', unzip=True)

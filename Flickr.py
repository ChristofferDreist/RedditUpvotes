#from ast import keyword
import flickrapi
import xmltodict
import pandas as pd
import datetime as dt
import os
import requests

api_key = u'370adf92054fe18ff1bbd0b1db83268f'
api_secret = u'1ccb478aead000d6'

flickr = flickrapi.FlickrAPI(api_key, api_secret, cache = True)

# Create empty data frame
posts_df = pd.DataFrame()

#with open("NatureTags.txt") as tags_file:
#    tags = [line.strip() for line in tags_file]


# Search parameters
keywords = 'nature, outdoors, landscape, wildlife, adventure, sailing, climbing, fishing, hiking' # Photos with one or more of these keywords in title, description, or tags will be returned (can also be used to exclude results with -keyword)
start_time = int(dt.datetime.timestamp(dt.datetime.strptime('2004-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')))
end_time = int(dt.datetime.timestamp(dt.datetime.strptime('2022-08-01 00:00:00', '%Y-%m-%d %H:%M:%S'))) 
current_time = int(dt.datetime.timestamp(dt.datetime.now())) 

# Fill in search details for when saving data as csv
search_description = 'nature_outdoors_landscape_wildlife_adventure_sailing_climbing_fishing_hiking'
search_time = '20040101-20220801' 

# Handling the number of results returned in total and per page
total_pages = 1
posts_per_page = 10

# Loop through flickr_search pages and add posts from each page to data frame
#for tag in tags:
for i in range(total_pages):    
    obj = flickr.photos.search(tags = keywords,
                               tag_mode = 'any', # 'any' -> OR combination of tags, 'all' -> AND combination of tags
                               min_upload_date = start_time, # min_taken_date -> start time based on when photo was taken instead of post time
                               max_upload_date = current_time, # max_taken_date -> end time based on when photo was taken instead of post time
                               media = 'all', # specify media type to only 'photos' or only 'videos' -> default is 'all'
                               has_geo = 1, # 1 = only photos that have been geotaggged, 0 = only photos that have not been geotagged
                               geo_context = 2, # 0 = not defined, 1 = indoors, 2 = outdoors 
                               extras = 'url_z, date_upload, date_taken, geo, media, machine_tags, tags, views, description, count_comments, count_faves, owner_name', # extra information returned for each item
                               per_page = posts_per_page,
                               page = i,
                               sort = "relevance",
                               format = 'rest')                       
    photos_dict = xmltodict.parse(obj)
    pics = photos_dict['rsp']['photos']['photo']
    new_page_df = pd.DataFrame.from_dict(pics)
    posts_df = pd.concat([posts_df, new_page_df])

# Set display options for pandas dataframe -> only a specified number of columns and rows are printed 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_row', 25)

# Preview sample of posts data
print(posts_df)

# Get list of all column names
posts_df.columns
posts_df.dtypes

# Remove unwanted columns
posts_df.drop(columns = ['@datetakengranularity', '@datetakenunknown', '@isfriend' , '@isfamily', '@geo_is_public', '@geo_is_contact', '@geo_is_friend', '@geo_is_family'], inplace = True)

# Rename columns using a dictionary -> 'Old name' : 'New name'
column_names = {'@id' : 'PhotoID',
                '@owner' : 'AuthorID',
                '@ownername' : 'AuthorName',
                '@secret' : 'Secret',
                '@server' : 'Server',
                '@farm' : 'Farm',
                '@title' : 'Title',
                '@ispublic' : 'IsPublic',
                '@dateupload' : 'DateUploaded', 
                '@datetaken' : 'DateTaken',
                '@views' : 'Views', 
                '@count_faves' : 'Faves',
                '@count_comments' : 'Comments',
                '@tags' : 'Tags',
                '@machine_tags' : 'MachineTags', 
                '@latitude' : 'Latitude', 
                '@longitude' : 'Longitude', 
                '@accuracy' : 'Accuracy', 
                '@context' : 'GeoContext',
                '@place_id' : 'PlaceID', #Flickr place id
                '@woeid' : 'WoeID', #32-bit identifier that uniquely represents spatial entities
                '@media' : 'Media', #MediaType
                '@media_status' : 'MediaStatus', 
                '@url_z' : 'URL',
                '@height_z' : 'PhotoHeight', 
                '@width_z' : 'PhotoWidth', 
                'description' : 'Description'
                }

posts_df.rename(columns = column_names, inplace = True)

# Check to see if columns have been renamed
posts_df.columns

# Reorder columns
posts_df = posts_df[['PhotoID', 'AuthorID', 'AuthorName', 'Title', 'Description', 'DateUploaded', 'DateTaken', 'URL', 'Views', 'Faves', 'Comments', 'Tags', 'MachineTags', 'Latitude',
                     'Longitude', 'Accuracy', 'GeoContext', 'PlaceID', 'WoeID', 'Secret', 'Server', 'Farm',  'IsPublic', 'Media', 'MediaStatus', 'PhotoHeight', 'PhotoWidth']]

print(posts_df)

# Convert TimeStamp from Unix to UTC
posts_df['DateUploaded'] = pd.to_datetime(posts_df['DateUploaded'], utc=True, unit='s')

# Check sample
print(posts_df['URL'])

# Save data frame as CSV
filename = 'C:/Users/acali/OneDrive - Danmarks Tekniske Universitet/Code/Flickr_'+ search_description + '_' + search_time + '.csv'
posts_df.to_csv(filename, header=True, index=False, columns=list(posts_df.axes[1]))

#Save images from URL in data frame
root_folder = 'C:/Users/acali/OneDrive - Danmarks Tekniske Universitet/Data/Flickr/'

def download(row):
   filename = root_folder + row['PhotoID'] + '.jpg'

   # create folder if it doesn't exist
   os.makedirs(os.path.dirname(filename), exist_ok = True)

   url = row.URL
   print(f"Downloading {url} to {filename}")
   r = requests.get(url, allow_redirects=True)
   with open(filename, 'wb') as f:
       f.write(r.content)

posts_df.apply(download, axis=1)
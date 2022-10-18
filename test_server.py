import os
import numpy as np
import pandas as pd
import json
from retinaface import RetinaFace
import cv2
from deepface import DeepFace

################### 1. Get the list of .json file name and paths


## This gets the current directory
pathFile = os.path.abspath(os.getcwd())
#print(pathFile)

## This defines the subdirectory to take into account


## ------->
subdir = '/crowdtangle-datacollection/facebook/'

subdir2 = '/crowdtangle-datacollection/instagram/'

tmp = [subdir, subdir2]

## This is the full directory where we do the analyses
listDir = os.listdir(pathFile + subdir)
listDir2 = os.listdir(pathFile + subdir2)

listDir = np.array(listDir)
listDir2 = np.array(listDir2)

listDir = np.append(listDir, listDir2)


## This defines objects that will be filles in the first loop. 

# Full paths to json files
pathToFiles = []

# Name of the json files
onlyFile = []


# Defines them as array
onlyFile = np.array(onlyFile)

pathToFiles = np.array(pathToFiles)

# Loops through each folder and each file in folder
for i in range(0, len(listDir)):
    if listDir[i][-5:] == '.json':
        files = os.listdir(pathFile + subdir + listDir[i])

    for j in range(0, len(files)):
        if files[j][-5:] == '.json':
            pathToFile = pathFile + subdir + listDir[i] + '/' + files[j]
            onlyFile = np.append(onlyFile, files[j][:-5])
            pathToFiles = np.append(pathToFiles, pathToFile)



dataSwissCanadianGerman = pd.read_csv('data_politicians_UK_CA_CH_DE.csv')

#dataImageJSON = pd.read_csv('Data collection/filesharing_political_science/data/crowdtangle/id_hash_mapping.csv')

def find_ref_image(dataSwissCanadianGerman, json_path):
    ## loads the json file
    data_json = json.load(open(json_path))

    ## Gets the account type and handle
    account_type = data_json['account']['accountType']
    
    account_handle = data_json['account']['handle']


    ##print(account_handle)
    if (account_type=='facebook_page'):
        try:
                
            img_url = dataSwissCanadianGerman[dataSwissCanadianGerman['facebook_id']==account_handle]['image_url']
            img_index = dataSwissCanadianGerman[dataSwissCanadianGerman['facebook_id']==account_handle].index.values.astype(int)[0]
            return(img_url[img_index])
        except:
            pass

    else:
        try:
            img_url = dataSwissCanadianGerman[dataSwissCanadianGerman['instagram_id']==account_handle]['image_url']
            img_index = dataSwissCanadianGerman[dataSwissCanadianGerman['instagram_id']==account_handle].index.values.astype(int)[0]
            return(img_url[img_index])
        except:
            pass
  
def find_ref_image_leader1(dataSwissCanadianGerman, json_path):
    ## loads the json file
    data_json = json.load(open(json_path))

    ## Gets the account type and handle
    account_type = data_json['account']['accountType']
    
    account_handle = data_json['account']['handle']


    ##print(account_handle)
    if (account_type=='facebook_page'):
        try:
                
            img_url = dataSwissCanadianGerman[dataSwissCanadianGerman['facebook_id']==account_handle]['image_url_party_leader1']
            img_index = dataSwissCanadianGerman[dataSwissCanadianGerman['facebook_id']==account_handle].index.values.astype(int)[0]
            return(img_url[img_index])
        except:
            pass

    else:
        try:
            img_url = dataSwissCanadianGerman[dataSwissCanadianGerman['instagram_id']==account_handle]['image_url_party_leader1']
            img_index = dataSwissCanadianGerman[dataSwissCanadianGerman['instagram_id']==account_handle].index.values.astype(int)[0]
            return(img_url[img_index])
        except:
            pass

def find_ref_image_leader2(dataSwissCanadianGerman, json_path):
    ## loads the json file
    data_json = json.load(open(json_path))

    ## Gets the account type and handle
    account_type = data_json['account']['accountType']
    
    account_handle = data_json['account']['handle']


    ##print(account_handle)
    if (account_type=='facebook_page'):
        try:
                
            img_url = dataSwissCanadianGerman[dataSwissCanadianGerman['facebook_id']==account_handle]['image_url_party_leader2']
            img_index = dataSwissCanadianGerman[dataSwissCanadianGerman['facebook_id']==account_handle].index.values.astype(int)[0]
            return(img_url[img_index])
        except:
            pass

    else:
        try:
            img_url = dataSwissCanadianGerman[dataSwissCanadianGerman['instagram_id']==account_handle]['image_url_party_leader2']
            img_index = dataSwissCanadianGerman[dataSwissCanadianGerman['instagram_id']==account_handle].index.values.astype(int)[0]
            return(img_url[img_index])
        except:
            pass


def get_largest_face (resp):

    sizeArray = []

    for key in resp:
        identity = resp[key]
        facial_area = identity["facial_area"]
        sqr_facial_area = (facial_area[2] - facial_area[0]) * (facial_area[3] - facial_area[1])
        sizeArray.append(sqr_facial_area)
    
    return(max(sizeArray))


#### Here change the function
#def get_image_url (dataImageJSON, json_path):
    try:
        url_img = dataImageJSON[dataImageJSON['json_ref']==json_path[:-5]]['img_ref']
        url_index = dataImageJSON[dataImageJSON['json_ref']==json_path[:-5]].index.values.astype(int)[0]
        return(url_img[url_index])
    except:
        pass

def recogn_pol (imgfaceurl, imgrefurl):
    verification = DeepFace.verify(img1_path = "Pol_data_image_ref/"+imgrefurl, img2_path = imgfaceurl, enforce_detection = False)
    #recognition = DeepFace.verify(img_path = "Data collection/Pol_data_image_ref/"+imgrefurl, db_path = dirRef)
    return(verification)

def get_image_faces (imgurl, imgrefurl, json_path, imgurlrefleader1, imgurlrefleader2):

    ## loads the json file
    data_json = json.load(open(json_path))
    
    account_type = data_json['account']['accountType']
    
    account_handle = data_json['account']['handle']

    account_id = data_json['account']['id']
    
    platform = data_json['account']['platform']
    
    countSubscriber = data_json['account']['subscriberCount']
    try:
        caption = data_json['caption']
    except:
        caption = ''
    
    datePost = data_json['date']
    
    try:
        description = data_json['description']
    except:
        description = ''
    try:
        message = data_json['message']
    except:
        message = ''
    
    comments = data_json['statistics']['actual']['commentCount']
    if platform == 'Facebook':
        favoriteCount = ''
        like = data_json['statistics']['actual']['likeCount']
        
        share = data_json['statistics']['actual']['shareCount']
        
        WowCount = data_json['statistics']['actual']['wowCount']
        HahaCount = data_json['statistics']['actual']['hahaCount']
        SadCount = data_json['statistics']['actual']['sadCount']
        AngryCount = data_json['statistics']['actual']['angryCount']
        ThankfulCount = data_json['statistics']['actual']['thankfulCount']
        CareCount = data_json['statistics']['actual']['careCount']
        LoveCount = data_json['statistics']['actual']['loveCount']
        

    if platform == 'Instagram':
        favoriteCount = data_json['statistics']['actual']['favoriteCount']
        like = ''
        
        share = ''
        
        WowCount = ''
        HahaCount = ''
        SadCount = ''
        AngryCount = ''
        ThankfulCount = ''
        CareCount = ''
        LoveCount = ''

    
    data = {
                    'imgID': [], ## URL of the image
                    'account_type': [],
                    'account_handle': [],
                    'account_id': [],
                    'platform': [],
                    'countSubscriber': [],
                    'caption': [],
                    'datePost': [],
                    'description': [],
                    'like': [],
                    'share': [],
                    'comments': [],
                    'WowCount': [],
                    'HahaCount': [],
                    'SadCount': [],
                    'AngryCount': [],
                    'ThankfulCount': [],
                    'CareCount': [],
                    'LoveCount': [],
                    'FavoriteCount': [],
                    'message': [],
                    'FaceTot': [],
                    'FaceNum': [], ## Face number in the image
                    'overalFaces': [],
                    'FaceArea': [], ## Area of the face in pxel square
                    'FacePos0': [], ## First coordinate of the face in the picture
                    'FacePos1': [], ## Second coordinate of the face in the picture
                    'FacePos2': [], ## Third coordinate of the face in the picture
                    'FacePos3': [], ## Fourth coordinate of the face in the picture
                    'FaceAge': [], ## Age of the face as predicted by deepface
                    'FaceGender': [], ## Gender of the face as predicted by deepface
                    'FaceEmotionAngry': [], ## Probability the face shows angry emotion
                    'FaceEmotionDisgust': [], ## Probability the face shows angry emotion
                    'FaceEmotionFear': [], ## Probability the face shows angry emotion
                    'FaceEmotionHappy': [], ## Probability the face shows angry emotion
                    'FaceEmotionSad': [], ## Probability the face shows angry emotion
                    'FaceEmotionSurprise': [], ## Probability the face shows angry emotion
                    'FaceEmotionNeutral': [], ## Probability the face shows angry emotion
                    'FaceEmotionDominant': [], ## Dominant emotion in the face
                    'raceAsian': [],
                    'raceIndian': [],
                    'raceBlack': [],
                    'raceMiddleEastern': [],
                    'raceWhite': [],
                    'raceLatinoHispanic': [],
                    'dominantRace': [],
                    'isPol': [],
                    'recognitionDistance': [],
                    'image_ref_url': [],
                    'is_leader1': [],
                    'recognitionDistance_leader1': [],
                    'image_ref_url_leader1': [],
                    'is_leader2': [],
                    'recognitionDistance_leader2': [],
                    'image_ref_url_leader2': []


                    ## Possible to add information about the race of the face.
                }

    df_marks = pd.DataFrame(data)

    try:            
        os.mkdir(json_path[:-5])
    except:
        pass
    try:
        
        #os.mkdir('Data collection/filesharing_political_science/data/crowdtangle/data face2/'+str(imgurl))


        filepath = json_path[:-5] + '.jpg'
        #print(filepath)
        ## Reads the image 
        #cv2.imshow(filepath)
        img = cv2.imread(filepath)
        #print(img)
        dimensions = img.shape
        #print(dimensions)
        ## Gets the faces
        resp = RetinaFace.detect_faces(filepath, threshold = 0.1)
        FaceTot = len(resp)
        max_size = get_largest_face(resp)
        #print(max_size)
        

        j = 0

        for key in resp:

            identity = resp[key]

            #---------------------
            # This computes the confidence score. Can be used to filter faces in pictures
            # confidence = identity["score"]

            # This gives the facial area
            facial_area = identity["facial_area"]

            sqr_facial_area = (facial_area[2] - facial_area[0]) * (facial_area[3] - facial_area[1])

            if (sqr_facial_area> 0.5 * max_size):
                # Defines the square of faces

                facial_area1 = facial_area[1]-50 if facial_area[1]-50>=0 else 0
                facial_area3 = facial_area[3]+50 if facial_area[3]+50<=dimensions[0] else dimensions[0]
                facial_area0 = facial_area[0]-50 if facial_area[0]-50>=0 else 0
                facial_area2 = facial_area[2]+50 if facial_area[2]+50<=dimensions[1] else dimensions[1]

                cropped_image = img[facial_area1:facial_area3, facial_area0:facial_area2]
            
                # Saves the faaces as original images
                cv2.imwrite(json_path[:-5] + '/img_face'+'_'+str(j)+'.jpg', cropped_image)
                #plt.imshow(cropped_image[:, :, ::-1])
                #plt.axis('off')
                #plt.show()

                analysis = DeepFace.analyze(json_path[:-5] + '/img_face'+'_'+str(j)+'.jpg', actions = ["age", "gender", "emotion", "race"], detector_backend = 'retinaface')

                try:
                    if type(imgrefurl) == str:
                        isRecogn = recogn_pol(json_path[:-5] + '/img_face'+'_'+str(j)+'.jpg', imgrefurl)
                        print(isRecogn)
                        isPol = isRecogn['verified']
                        recognitionDistance = isRecogn['distance']
                        image_ref_url = imgrefurl
                        
                    else:
                        isPol = ''
                        recognitionDistance = ''
                        image_ref_url = ''
                except:
                    isPol = ''
                    recognitionDistance = ''
                    image_ref_url = ''
                
                
                try:
                    if type(imgurlrefleader1)==str:
                        isRecogn = recogn_pol(json_path[:-5] + '/img_face'+'_'+str(j)+'.jpg', img_url_ref_leader1)
                        print(isRecogn)
                        is_leader1 = isRecogn['verified']
                        recognitionDistance_leader1 = isRecogn['distance']
                        image_ref_url_leader1 = imgurlrefleader1
                        
                    else:
                        is_leader1 = ''
                        recognitionDistance_leader1 = ''
                        image_ref_url_leader1 = ''
                except:
                    is_leader1 = ''
                    recognitionDistance_leader1 = ''
                    image_ref_url_leader1 = ''

                try:

                    if type(imgurlrefleader2)==str:
                        isRecogn = recogn_pol(json_path[:-5] + '/img_face'+'_'+str(j)+'.jpg', img_url_ref_leader2)
                        print(isRecogn)
                        is_leader2 = isRecogn['verified']
                        recognitionDistance_leader2 = isRecogn['distance']
                        image_ref_url_leader2 = imgurlrefleader2
                        
                    else:
                        is_leader2 = ''
                        recognitionDistance_leader2 = ''
                        image_ref_url_leader2 = ''
                except:
                    is_leader2 = ''
                    recognitionDistance_leader2 = ''
                    image_ref_url_leader2 = ''

                newrow = {
                            'imgID': imgurl, ## URL of the image
                            'account_type': account_type,
                            'account_handle': account_handle,
                            'account_id': account_id,
                            'platform': platform,
                            'countSubscriber': countSubscriber,
                            'caption': caption,
                            'datePost': datePost,
                            'description': description,
                            'like': like,
                            'share': share,
                            'comments': comments,
                            'WowCount': WowCount,
                            'HahaCount': HahaCount,
                            'SadCount': SadCount,
                            'AngryCount': AngryCount,
                            'ThankfulCount': ThankfulCount,
                            'CareCount': CareCount,
                            'LoveCount': LoveCount,
                            'FavoriteCount': favoriteCount,
                            'message': message,
                            'FaceTot': FaceTot,
                            'FaceNum': j, ## Face number in the image
                            'FaceArea': sqr_facial_area, ## Area of the face in pxel square
                            'FacePos0': facial_area[0], ## First coordinate of the face in the picture
                            'FacePos1': facial_area[1], ## Second coordinate of the face in the picture
                            'FacePos2': facial_area[2], ## Third coordinate of the face in the picture
                            'FacePos3': facial_area[3], ## Fourth coordinate of the face in the picture
                            'FaceAge': analysis['age'], ## Age of the face as predicted by deepface
                            'FaceGender': analysis['gender'], ## Gender of the face as predicted by deepface
                            'FaceEmotionAngry': analysis['emotion']['angry'], ## Probability the face shows angry emotion
                            'FaceEmotionDisgust': analysis['emotion']['disgust'], ## Probability the face shows angry emotion
                            'FaceEmotionFear': analysis['emotion']['fear'], ## Probability the face shows angry emotion
                            'FaceEmotionHappy': analysis['emotion']['happy'], ## Probability the face shows angry emotion
                            'FaceEmotionSad': analysis['emotion']['sad'], ## Probability the face shows angry emotion
                            'FaceEmotionSurprise': analysis['emotion']['surprise'], ## Probability the face shows angry emotion
                            'FaceEmotionNeutral': analysis['emotion']['neutral'], ## Probability the face shows angry emotion
                            'FaceEmotionDominant': analysis['dominant_emotion'], ## Dominant emotion in the face
                            'raceAsian': analysis['race']['asian'],
                            'raceIndian': analysis['race']['indian'],
                            'raceBlack': analysis['race']['black'],
                            'raceMiddleEastern': analysis['race']['middle eastern'],
                            'raceWhite': analysis['race']['white'],
                            'raceLatinoHispanic': analysis['race']['latino hispanic'],
                            'dominantRace': analysis['dominant_race'],
                            'isPol': isPol,
                            'recognitionDistance': recognitionDistance,
                            'image_ref_url': image_ref_url,
                            'is_leader1': is_leader1,
                            'recognitionDistance_leader1': recognitionDistance_leader1,
                            'image_ref_url_leader1': image_ref_url_leader1,
                            'is_leader2': is_leader2,
                            'recognitionDistance_leader2': recognitionDistance_leader2,
                            'image_ref_url_leader2': image_ref_url_leader2
                            ## Possible to add information about the race of the face.
                        }

                print(newrow)

                df_marks = df_marks.append(newrow, ignore_index=True)
                
                j = j+1
            
                

        
        
        return(df_marks)
    except:
        newrow = {
                            'imgID': imgurl, ## URL of the image
                            'account_type': account_type,
                            'account_handle': account_handle,
                            'account_id': account_id,
                            'platform': platform,
                            'countSubscriber': countSubscriber,
                            'caption': caption,
                            'datePost': datePost,
                            'description': description,
                            'like': like,
                            'share': share,
                            'comments': comments,
                            'WowCount': WowCount,
                            'HahaCount': HahaCount,
                            'SadCount': SadCount,
                            'AngryCount': AngryCount,
                            'ThankfulCount': ThankfulCount,
                            'CareCount': CareCount,
                            'LoveCount': LoveCount,
                            'FavoriteCount': favoriteCount,
                            'message': message,
                            'FaceTot': '',
                            'FaceNum': '', ## Face number in the image
                            'FaceArea': '', ## Area of the face in pxel square
                            'FacePos0': '', ## First coordinate of the face in the picture
                            'FacePos1': '', ## Second coordinate of the face in the picture
                            'FacePos2': '', ## Third coordinate of the face in the picture
                            'FacePos3': '', ## Fourth coordinate of the face in the picture
                            'FaceAge': '', ## Age of the face as predicted by deepface
                            'FaceGender': '', ## Gender of the face as predicted by deepface
                            'FaceEmotionAngry': '', ## Probability the face shows angry emotion
                            'FaceEmotionDisgust': '', ## Probability the face shows angry emotion
                            'FaceEmotionFear': '', ## Probability the face shows angry emotion
                            'FaceEmotionHappy': '', ## Probability the face shows angry emotion
                            'FaceEmotionSad': '', ## Probability the face shows angry emotion
                            'FaceEmotionSurprise': '', ## Probability the face shows angry emotion
                            'FaceEmotionNeutral': '', ## Probability the face shows angry emotion
                            'FaceEmotionDominant': '', ## Dominant emotion in the face
                            'raceAsian': '',
                            'raceIndian': '',
                            'raceBlack': '',
                            'raceMiddleEastern': '',
                            'raceWhite': '',
                            'raceLatinoHispanic': '',
                            'dominantRace': '',
                            'isPol': '',
                            'recognitionDistance': '',
                            'image_ref_url': '',
                            'is_leader1': '',
                            'recognitionDistance_leader1': '',
                            'image_ref_url_leader1': '',
                            'is_leader2': '',
                            'recognitionDistance_leader2': '',
                            'image_ref_url_leader2': ''
                            ## Possible to add information about the race of the face.
                        }

        df_marks = df_marks.append(newrow, ignore_index=True)
    



#data_faces = pd.read_csv('Data collection 2/out_data_test4.csv')

for i in list(range(0, len(pathToFiles))):

        print(pathToFiles[i])
        img_ref_url = find_ref_image(dataSwissCanadianGerman, pathToFiles[i])
        #print(img_ref_url)
        #img_url = get_image_url(dataImageJSON, pathToFiles[i])
        #print(img_url)
        img_url_ref_leader1 = find_ref_image_leader1(dataSwissCanadianGerman, pathToFiles[i])
        #print(len(img_url_ref_leader1))
        img_url_ref_leader2 = find_ref_image_leader2(dataSwissCanadianGerman, pathToFiles[i])
        #print(img_url_ref_leader2)
        if i == 0:

            data_faces = get_image_faces(pathToFiles[i][:-5] + ".jpg",img_ref_url,pathToFiles[i], img_url_ref_leader1, img_url_ref_leader2)
            
        
        if i>0:
            dta_faces = get_image_faces(pathToFiles[i][:-5] + ".jpg",img_ref_url,pathToFiles[i], img_url_ref_leader1, img_url_ref_leader2)
            
            data_faces = pd.concat([data_faces,dta_faces])
            data_faces.to_csv('out_data_test.csv') 
        print(i)
        #print(img_ref_url)
        #print(img_url)
        
        

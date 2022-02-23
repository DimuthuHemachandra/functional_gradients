"""
Purpose : To extract PD demographic features from tables stored in Data/demogrphic

Inputs  : CSV files for demographic data

Outputs : Data frames with demographic data sorted according to the data provided


"""
import numpy as np 
import pandas as pd


PD_Feature_data_path = "../cfg/demographic/PD_Features.csv"
summary_data_path = "../cfg/demographic/Baseline_Data_Summary.csv"
PD_medication_data_path = "../cfg/demographic/Use_of_PD_Medication.csv"

def get_demographic(Data_df, PD_demographic_df,feature,session, sub_ID):
  """This method reads a dataframe (PD data) and a csv file that had demographic data and returns a dataframe 
  including a specified feature (Example = "DOMSIDE" 

  Data_df: Data frame with diffparc PD results
  PD_feature_data_path: Path to a csv file with demographic data
  feature: Feature name from the csv file (eg: "DOMSIDE")
  session: String with the session name (eg: "Baseline")
  sub_ID: String with the name of the column with subject IDs

  returns a dataframe with PD diffpac data including the feature values"""


  #PD_demographic_df = pd.read_csv(PD_demographic_data_path)

  PD_subj_list= Data_df["subj"] # Getting the subject list

  subj_number=[]

  #Stripping the session and getting subject ID
  for subjects in PD_subj_list:
    splitted_name=subjects.split('-')
    splitted_name=(splitted_name[1]).split('_')
    subj_number.append(int(splitted_name[0]))

  #Mathches subject numbers with Patient number (PATNO)
  PD_demographic_df=PD_demographic_df[PD_demographic_df[sub_ID].isin(subj_number)]

  ID_list = list(PD_demographic_df[sub_ID])

  subj = []
  #Adding bids features to PATNO
  for subject in ID_list:
    subject_new = "sub-"+str(subject)   #+"_ses-"+session
    subj.append(subject_new) 

  #Replacing 'PATNO' with bidsified subj names
  PD_demographic_df.drop(sub_ID, axis = 1, inplace = True)
  PD_demographic_df[sub_ID] = subj
  PD_demographic_df = PD_demographic_df.rename(columns={sub_ID: 'subj'})

  demographic_sorted = PD_demographic_df[["subj",feature]]

  df = pd.merge(demographic_sorted, Data_df, on='subj')

  return df


def get_side_affected(PD_data,session):
  """Reads a data frame from PD diffparc data and adds the affected side info into it
  PD_data: PD diffparc dataframe
  session: string (eg: "Baseline")

  return: data frame with affected side info"""

  #getting the dominant side (DOMSIDE) info
  PD_Feature_df = pd.read_csv(PD_Feature_data_path)
  feature_df = get_demographic(PD_data,PD_Feature_df,"DOMSIDE",session,'PATNO')

  #LEft side is 1 and right side is 2
  PD_left = feature_df.loc[feature_df['DOMSIDE'] == "1"]
  PD_right = feature_df.loc[feature_df['DOMSIDE'] == "2"]

  return PD_left,PD_right


def get_UPDRS(PD_data,UPDRS_Nu,session):

  PD_UPDRS_df = pd.read_csv(summary_data_path)
  df = get_demographic(PD_data,PD_UPDRS_df,UPDRS_Nu,session,'Subject Number')
  return df

def get_PD_medication(PD_data,data_col,session):

  PD_medication_df = pd.read_csv(PD_medication_data_path)
  if session == "Baseline":
    event_ID = "V01"
  if session == "Month12":
    event_ID = "V04"
  if session == "Month24":
    event_ID = "V06"
  PD_med_df = PD_medication_df.loc[PD_medication_df['EVENT_ID'] == event_ID]
  PD_med_df = get_demographic(PD_data,PD_med_df,data_col,session,'PATNO')

  PD_off_med_df = PD_med_df.loc[PD_med_df['PDMEDYN'] == 0]
  PD_on_med_df = PD_med_df.loc[PD_med_df['PDMEDYN'] == 1]

  return PD_med_df,PD_off_med_df,PD_on_med_df

def get_any_demo(df,info,session):
  #info: Column name from the summary table

  demo_df = pd.read_csv(summary_data_path)
  print(list(demo_df))
  df = get_demographic(df,demo_df,info,session,'Subject Number')
  print(list(df))
  return df
  
  



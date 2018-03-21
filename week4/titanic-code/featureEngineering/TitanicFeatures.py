import pandas as pd
import re

from auxiliary.frameWrangler import one_hot
from featureEngineering.Features import Features


# Does specific feature engineering for the Titanic task. All features are just added to the existing features
class TitanicFeatures(Features):

    def engineer_features(self, data, verbose=False):

        print ("Engineering features..")

        # Put features in bins with a numerical ID, to make it easier to train on
        data = binAge(data)
        data = binFares(data)

        # Create new features by combining existing ones
        data = combineAgePClass(data)

        # Engineer completely new features
        data = createTicketFirstChar(data)
        data = createFamilySize(data)
        data = createFamilySizeBinned(data)
        data = createIsAlone(data)
        data = createTitle(data)
        data = createTitleAlt(data)

        # cabin features
        # data = extractCabinFeatures(data)
        #
        # data = extractNameFeatures(data)
        data = createEmbarked(data)

        data = one_hot(data, ['Title'], drop_col=False)

        if verbose:
            print( "\n" + ('-' * 40) )
            print( " Data after feature engineering")
            print( '-' * 40)
            print( data.head() )

        # data = data.drop(['SibSp', 'Parch'], axis=1)
        print ("Features:")
        print (list(data))

        return data


# Bin Age and save it as a new feature
def binAge(data):
    interval = (0, 5, 12, 18, 25, 35, 60, 120)
    cats = [0,1,2,3,4,5,6]
    data["Age_cat"] = pd.cut(data.Age, interval, labels=cats)

    return data


# Put fares into 4 bins (see https://www.kaggle.com/polarhut/titanic-data-science-solutions)
def binFares(data):
    data['Fare_cat'] = data['Fare']
    data.loc[ data['Fare_cat'] <= 7.91, 'Fare_cat'] = 0
    data.loc[(data['Fare_cat'] > 7.91) & (data['Fare'] <= 14.454), 'Fare_cat'] = 1
    data.loc[(data['Fare_cat'] > 14.454) & (data['Fare'] <= 31), 'Fare_cat']   = 2
    data.loc[ data['Fare_cat'] > 31, 'Fare_cat'] = 3
    data['Fare_cat'] = data['Fare_cat'].astype(int)

    return data


# Create new class Pclass*Age
def combineAgePClass(data):
    data['Age*Class'] = data.Age_cat * data.Pclass
    # data.loc[:, ['Age*Class', 'Age_cat', 'Pclass']].head(10)

    return data

# Create a new feature from the first character of the Ticket
def createTicketFirstChar(data):
    data['Ticket_firstchar'] = data['Ticket']
    data.Ticket_firstchar = data.Ticket_firstchar.map(lambda x: x[0])
    data["Ticket_firstchar"].replace(['A', 'P', 'S', 'C', 'W', 'F', 'L', '1','2','3','4','5','6','7','8','9'], [10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9], inplace=True)

    return data


# Create new feature familysize= self + sibelings/spouse + Parents/children
def createFamilySize(data):
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    return data


# create new feature based on family size, but in categories
def createFamilySizeBinned(data):
    data['FamilySize_cat'] = data['FamilySize']
    data.loc[ data['FamilySize_cat'] <= 1, 'FamilySize_cat'] = 1
    data.loc[(data['FamilySize_cat'] >= 2) & (data['FamilySize_cat'] < 5), 'FamilySize_cat'] = 2
    data.loc[data['FamilySize_cat'] >= 5 , 'FamilySize_cat']   = 3
    # print ( data.groupby('FamilySize_cat')['PassengerId'].nunique() )

    return data


# create new feature which classifies people as alone or not
def createIsAlone(data):
    data['IsAlone'] = 0
    data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1

    return data

def createEmbarked(data):
    # one-hot encoding of the embarked feature
    data = one_hot(data, ['Embarked'], drop_col=False)

    return data

def extractCabinFeatures(data):
    # absence of cabin data might be associated with lower status
    data['hasCabinData'] = data["Cabin"].isnull().apply(lambda x: not x)

    # get the deck information
    data['Deck'] = data['Cabin'].str.slice(0,1)
    # replace N/A
    data['Deck'] = data['Deck'].fillna("N")

    # get the room information
    #  data['Room'] = \
    #  data['Cabin'].str.slice(1, 5).str.extract("([0-9]+)", expand=False).astype("float")
    # replace N/A
    #  data['Room'] = data['Room'].fillna(data["Room"].mean())

    # get a one-hot encoding of deck numbers
    data = one_hot(data, ['Deck'], drop_col=False)
    return data



# create a new feature which classifies people based on their title,
# with categories "Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5
def createTitle(data):
    # first extract the titles of the people
    data['Title'] = data.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
    # data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False) # only works in python 3
    data['Title'] = data['Title'].replace(['Lady', 'Countess','the Countess', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')

    # convert the title to a numerical ID and fill in empty rows
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    data['Title'] = data['Title'].map(title_mapping)
    data['Title'] = data['Title'].fillna(0)

    return data


# Create a new feature which classifies people based on their title, alternative to above
# with the categories "Officer": 1, "Royalty": 2, "Mrs": 3, "Miss": 4, "Mr": 5, "Master": 6
def createTitleAlt(data):
    data['Title_alt'] = data.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
    # data['Title_alt'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False) # only works in python 3
    Title_Dictionary = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Dr": "Officer",
        "Rev": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir": "Royalty",
        "the Countess": "Royalty",
        "Countess": "Royalty",
        "Dona": "Royalty",
        "Lady": "Royalty",
        "Mme": "Mrs",
        "Ms": "Mrs",
        "Mrs": "Mrs",
        "Mlle": "Miss",
        "Miss": "Miss",
        "Mr": "Mr",
        "Master": "Master"
    }
    data['Title_alt'] = data.Title_alt.map(Title_Dictionary)

    # convert the title to a numerical ID and fill in empty rows
    title_mapping = {"Officer": 1, "Royalty": 2, "Mrs": 3, "Miss": 4, "Mr": 5, "Master": 6}
    data['Title_alt'] = data['Title_alt'].map(title_mapping)
    data['Title_alt'] = data['Title_alt'].fillna(0)

    return data

def extractNameFeatures(data):
    # longer names might be associated with higher socio-economic status
    data['nameLength'] = data['Name'].apply(lambda x : len(x))

    return data

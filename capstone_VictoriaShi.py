#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 16:36:56 2022

@author: victoriashi
"""



import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


data = pd.read_csv('/Users/victoriashi/Documents/Courses/Intro to DS/Data Analysis Reports/movieReplicationSet.csv')

#%% Q1: What is the relationship between sensation seeking and movie experience?

#stack sensation seeking data and movie experience data
sensation = data.iloc[:, 400:420]
experience = data.iloc[:, 464:474]
sensationAndExperience = pd.concat([sensation,experience], axis = 1)
sensationAndExperience.dropna(inplace = True)

nonnanSensation = sensationAndExperience.iloc[:,0:20]
nonnanExperience = sensationAndExperience.iloc[:,20:]


#PCA: dimensionality reduction for sensation seeking
arrSensation = nonnanSensation.to_numpy()
print(arrSensation.shape)
plt.imshow(arrSensation, aspect='auto')
plt.xlabel('Sensation Seeking')
plt.ylabel('Viewers')
plt.colorbar()
plt.show()

corrMatrixSensation = np.corrcoef(arrSensation,rowvar=False)     
plt.imshow(corrMatrixSensation) 
plt.xlabel('Sensation Seeking')
plt.ylabel('Sensation Seeking')
plt.colorbar()
plt.show()  

zscoredSensation = stats.zscore(arrSensation)
pca = PCA().fit(zscoredSensation)
eigValsSensation = pca.explained_variance_
loadingsSensation = pca.components_
rotatedSensation = pca.fit_transform(zscoredSensation)

varExplainedSensation = eigValsSensation/sum(eigValsSensation)*100  
for ii in range(len(varExplainedSensation)):
    print(varExplainedSensation[ii].round(3))
    
numSensation = len(sensation.columns)
x = np.linspace(1,numSensation,numSensation)
plt.bar(x, eigValsSensation, color='gray')
plt.plot([0,numSensation],[1,1],color='orange') # Orange Kaiser criterion line for the fox
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.title('Screeplot for Sensation Seeking')
plt.show()

kaiserThreshold = 1
print('Number of factors selected by Kaiser criterion:', np.count_nonzero(eigValsSensation > kaiserThreshold))


dfSensationQuestions = pd.DataFrame(list(nonnanSensation.columns))
whichPrincipalComponent = 1 # Select and look at one factor at a time 
plt.bar(x,loadingsSensation[whichPrincipalComponent,:]*-1) # note: eigVecs multiplied by -1 because the direction is arbitrary
plt.xlabel('Sensation Seeking Question')
plt.ylabel('Loading')
plt.show() # Show bar plot
print(dfSensationQuestions.values)
#factor 1: organization / planning / sense of order (conservativeness)
#factor 3: horror
#factor 5: risk-preference
#factor 6: short-term excitement / impulse

plt.plot(rotatedSensation[:,0],rotatedSensation[:,1],'o',markersize=1)
plt.xlabel('Sense of Order')
plt.ylabel('Excitement/Risk/Impulse Seeking')
plt.show()

origDataNewCoordinates = pca.fit_transform(zscoredSensation)*-1
sensationX = np.column_stack((origDataNewCoordinates[:,0],origDataNewCoordinates[:,1]))


#PCA: dimensionality reduction for movie experience

arrExperience = nonnanExperience.to_numpy()
print(arrExperience.shape)
plt.imshow(arrExperience, aspect='auto')
plt.xlabel('Experience')
plt.ylabel('Viewers')
plt.colorbar()
plt.show()

corrMatrixExperience = np.corrcoef(arrExperience,rowvar=False)     
plt.imshow(corrMatrixExperience) 
plt.xlabel('Experience')
plt.ylabel('Experience')
plt.colorbar()
plt.show()  

zscoredExperience = stats.zscore(arrExperience)
pca = PCA().fit(zscoredExperience)
eigValsExperience = pca.explained_variance_
loadingsExperience = pca.components_
rotatedExperience = pca.fit_transform(zscoredExperience)

varExplainedExperience = eigValsExperience/sum(eigValsExperience)*100  
for ii in range(len(varExplainedExperience)):
    print(varExplainedExperience[ii].round(3))
    
numExperience = len(experience.columns)
x = np.linspace(1,numExperience,numExperience)
plt.bar(x, eigValsExperience, color='gray')
plt.plot([0,numExperience],[1,1],color='orange') # Orange Kaiser criterion line for the fox
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.title('Screeplot for Movie Experience')
plt.show()

kaiserThreshold = 1
print('Number of factors selected by Kaiser criterion:', np.count_nonzero(eigValsExperience > kaiserThreshold))

dfExperienceQuestions = pd.DataFrame(list(nonnanExperience.columns))
whichPrincipalComponent = 3 # Select and look at one factor at a time 
plt.bar(x,loadingsExperience[whichPrincipalComponent,:]*-1) # note: eigVecs multiplied by -1 because the direction is arbitrary
plt.xlabel('Experience Question')
plt.ylabel('Loading')
plt.show() # Show bar plot
print(dfExperienceQuestions.values)

#factor 1: empathy
#factor 2: emotional instability

sensationPrincipal = rotatedSensation[:,:6]
experiencePrincipal = rotatedExperience[:,:2]

sensationMean = np.mean(sensationPrincipal,axis=1)
experienceMean = np.mean(experiencePrincipal,axis=1)

corrSE = np.corrcoef(sensationMean,experienceMean)
print(corrSE)

plt.plot(sensationMean,experienceMean, 'o', markersize=4)
plt.xlabel('Sensation Seeking')
plt.ylabel('Movie Experience')
plt.show()


'''
origDataNewCoordinatesExperience = pca.fit_transform(zscoredExperience)*-1
experienceX = np.column_stack((origDataNewCoordinatesExperience[:,0],origDataNewCoordinatesExperience[:,1]))

plt.plot(sensationX,experienceX, 'o')
plt.xlabel('Sensation Seeking')
plt.ylabel('Movie Experience')
'''


'''
for i in range(1,10):
    x = i-0.5
    y = i-0.5
    
    PC1 = pd.DataFrame(loadingsSensation[:,:i])
    PC2 = pd.DataFrame(loadingsExperience[:,:i])
    
    finalCorr = pd.concat([PC1,PC2], axis=1)
    corrMatrixSE = finalCorr.corr()
    
    plt.plot(3,3,i)
    plt.imshow(corrMatrixSE,aspect='auto')
    
    plt.axvline(x=x,color='r')
    plt.axhline(y=y,color='r')
    
    plt.xlabel('Sensation Seeking', fontsize=8)
    plt.xlabel('Movie Experience', fontsize=8)
    
    plt.title(f'{i} PCs')
    
    plt.colorbar()
    plt.tight_layout()

print(corrMatrixSE)
'''

#%% Q2: Is there evidence of personality types based on the data of these research participants? If so,
#characterize these types both quantitatively and narratively.
#PCA: personality
personality = data.iloc[:, 420:464]
personality.dropna(inplace = True)
arrPersonality = personality.to_numpy()
print(arrPersonality.shape)
plt.imshow(arrPersonality, aspect='auto')
plt.xlabel('Personality')
plt.ylabel('Viewers')
plt.colorbar()
plt.show()

corrMatrixPersonality = np.corrcoef(arrPersonality,rowvar=False)     
plt.imshow(corrMatrixPersonality) 
plt.xlabel('Personality')
plt.ylabel('Personality')
plt.colorbar()
plt.show()  

zscoredPersonality = stats.zscore(arrPersonality)
pca = PCA().fit(zscoredPersonality)
eigValsPersonality = pca.explained_variance_
loadingsPersonality = pca.components_
rotatedPersonality = pca.fit_transform(zscoredPersonality)

varExplainedPersonality = eigValsPersonality/sum(eigValsPersonality)*100  
for ii in range(len(varExplainedPersonality)):
    print(varExplainedPersonality[ii].round(3))
    
numPersonality = len(personality.columns)
x = np.linspace(1,numPersonality,numPersonality)
plt.bar(x, eigValsPersonality, color='gray')
plt.plot([0,numPersonality],[1,1],color='orange') # Orange Kaiser criterion line for the fox
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.title('Screeplot for Personality')
plt.show()

kaiserThreshold = 1
print('Number of factors selected by Kaiser criterion:', np.count_nonzero(eigValsPersonality > kaiserThreshold))

dfPersonalityQuestions = pd.DataFrame(list(personality.columns))
whichPrincipalComponent = 1 # Select and look at one factor at a time 
plt.bar(x,loadingsPersonality[whichPrincipalComponent,:]*-1) # note: eigVecs multiplied by -1 because the direction is arbitrary
plt.xlabel('Personality Question')
plt.ylabel('Loading')
plt.show() # Show bar plot
print(dfPersonalityQuestions.values)
#factor1: enthusiasm / extroversion
#factor2: active mind
#factor3: introversion / quietness
#factor4: perseverence / patience

personalityX = rotatedPersonality[:,:8]

numClusters = 9
sSum = np.empty([numClusters,1])*np.NaN
for ii in range(2, numClusters+2): # Loop through each cluster (from 2 to 10)
    kMeans = KMeans(n_clusters = int(ii)).fit(personalityX) # compute kmeans using scikit
    cId = kMeans.labels_ # vector of cluster IDs that the row belongs to
    cCoords = kMeans.cluster_centers_ # coordinate location for center of each cluster
    s = silhouette_samples(personalityX,cId) # compute the mean silhouette coefficient of all samples
    sSum[ii-2] = sum(s) # take the sum
    # Plot data:
    plt.subplot(3,3,ii-1) 
    plt.hist(s,bins=20) 
    plt.xlim(-0.2,1)
    plt.ylim(0,250)
    plt.xlabel('Silhouette score')
    plt.ylabel('Count')
    plt.title('Sum: {}'.format(int(sSum[ii-2]))) # sum rounded to nearest integer
    plt.tight_layout() # adjusts subplot 

plt.plot(np.linspace(2,10,9),sSum)
plt.xlabel('Number of clusters')
plt.ylabel('Sum of silhouette scores')
plt.show()

numClusters = 2
kMeans = KMeans(n_clusters = numClusters).fit(personalityX)
cId = kMeans.labels_
cCoords = kMeans.cluster_centers_
for ii in range (numClusters):
    plotIndex = np.argwhere(cId == int(ii))
    plt.plot(personalityX[plotIndex,0], personalityX[plotIndex,1],'o',markersize=1)
    plt.plot(cCoords[int(ii-1),0],cCoords[int(ii-1),1],'o',markersize=5,color='black')
    plt.xlabel('Enthusiasm / Extroversion')
    plt.ylabel('Active Mind')


#%% Q3: Are movies that are more popular rated higher than movies that are less popular?

rating = data.iloc[:,0:400]
#arrRating = rating.to_numpy()
#arrNans=np.empty([1,400])
#arrNans[:] = np.NaN

           
nans=rating.isnull().sum(axis=0)
#nans.sort_values(axis=0, ascending=True, inplace=True)
means=rating.mean(axis=0)

plt.plot(nans,means,'o')
plt.xlabel('Number of NaNs (unpopularity)')
plt.ylabel('Ratings')
plt.title('Correlation Between (Un)Popularity and Ratings')

rho = stats.spearmanr(nans,means)
print(rho.correlation)
actualRho = rho.correlation

plt.hist(nans) #the data is left-skewed
plt.hist(nans**3)
plt.hist(means)
#paired (dependent) t-test
arrNansCube = (nans.to_numpy())**3
arrMeans = means.to_numpy()
t1,p1 = stats.ttest_rel(arrNansCube,arrMeans)
print("p-value for dependent t-test of popularity (number of nans) and ratings (mean):", p1)

#%% Q4: Is enjoyment of ‘Shrek (2001)’ gendered, i.e. do male and female viewers rate it differently?
gender = data.iloc[:, 474]
dfGender = gender.to_frame()
shrekList = ['Shrek (2001)']
shrekRating = rating[shrekList]
shrekRatingGender = pd.concat([shrekRating, dfGender],axis=1)
shrekRatingGender.dropna(inplace = True)

#sort the dataframe according to gender
shrekRatingGender.sort_values(by=['Gender identity (1 = female; 2 = male; 3 = self-described)'],inplace=True)
arrShrekRatingGender = shrekRatingGender.to_numpy()

#put different genders into different arrays
femaleShrek= arrShrekRatingGender[0:743,0]
maleShrek = arrShrekRatingGender[743:984,0]
otherShrek = arrShrekRatingGender[984:,0]

#null hypothesis: the ratings have no difference (all genders rate Shrek equally)
#first take a look at the distributions of ratings for different genders
plt.hist(femaleShrek[:],bins=9)
plt.title('Distribution of Ratings of Female Viewers')
plt.xlabel('Ratings')
plt.ylabel('Female Viewers')
plt.hist(maleShrek[:],bins=9)
plt.title('Distribution of Ratings of Male Viewers')
plt.xlabel('Ratings')
plt.ylabel('Male Viewers')
plt.hist(otherShrek[:],bins=9)
plt.title('Distribution of Ratings of Self-identified-gendered Viewers')
plt.xlabel('Ratings')
plt.ylabel('Self-identified-gendered Viewers')
#clearly, the data isn't normally distributed, so a non-parametric test may be more appropriate
u1,up1 = stats.mannwhitneyu(femaleShrek[:],maleShrek[:]) #Mann-Whitney U test: female and male
print("Test statistic U for Mann-Whitney U test of female and male ratings:", u1)
print("p-value for Mann-Whitney U test of female and male ratings:", up1)
print()
u2,up2 = stats.mannwhitneyu(femaleShrek[:],otherShrek[:]) #Mann-Whitney U test: female and self-described
print("Test statistic U for Mann-Whitney U test of female and self-described gender ratings:", u2)
print("p-value for Mann-Whitney U test of female and self-described gender ratings:", up2)
print()
u3,up3 = stats.mannwhitneyu(maleShrek[:],otherShrek[:]) #Mann-Whitney U test: male and self-described
print("Test statistic U for Mann-Whitney U test of male and self-described gender ratings:", u3)
print("p-value for Mann-Whitney U test of male and self-described gender ratings:", up3)
print()

#%% Q5: Do people who are only children enjoy ‘The Lion King (1994)’ more than people with siblings?
onlychild = data.iloc[:, 475]
dfChild = onlychild.to_frame()
lionKingList = ['The Lion King (1994)']
lionKingRating = rating[lionKingList]
lionKingRatingChild = pd.concat([lionKingRating, dfChild], axis = 1)
lionKingRatingChild.dropna(inplace = True)
lionKingRatingChild.sort_values(by=['Are you an only child? (1: Yes; 0: No; -1: Did not respond)'],inplace=True)
arrlionKingRatingChild = lionKingRatingChild.to_numpy()

# separate the participants who have siblings from those who are only children
siblingLionKing = arrlionKingRatingChild[10:786,0]
onlyLionKing = arrlionKingRatingChild[786:,0]

#null hypothesis: the ratings have no difference no matter people are only children or not
plt.hist(siblingLionKing[:],bins=9)
plt.title('Distribution of Ratings for Lion King of Viewers Who Have Siblings')
plt.xlabel('Ratings')
plt.ylabel('Viewers with Siblings')

plt.hist(onlyLionKing[:],bins=9)
plt.title('Distribution of Ratings for Lion King of Viewers Who are Only Child')
plt.xlabel('Ratings')
plt.ylabel('Viewers Who are Only Child')

u4, up4 = stats.mannwhitneyu(onlyLionKing[:],siblingLionKing[:]) #Mann-Whitney U test: only children and with siblings
print("- Test statistic U for Mann-Whitney U test of ratings of viewers who are only child and viewers who have siblings:", u4)
print("- p-value for Mann-Whitney U test of ratings of viewers who are only child and viewers who have siblings:", up4)

#%% Q6: Do people who like to watch movies socially enjoy ‘The Wolf of Wall Street (2013)’ more than those who prefer to watch them alone?
socialAlone = data.iloc[:, 476]
dfSocialAlone = socialAlone.to_frame()
WWSList = ['The Wolf of Wall Street (2013)']
WWSRating = rating[WWSList]
WWSRatingSocial = pd.concat([WWSRating, dfSocialAlone], axis = 1)
WWSRatingSocial.dropna(inplace = True)
WWSRatingSocial.sort_values(by=['Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)'],inplace=True)
arrWWSRatingSocial = WWSRatingSocial.to_numpy()

socialWWS = arrWWSRatingSocial[4:274,0]
aloneWWS = arrWWSRatingSocial[274:,0]

plt.hist(socialWWS[:],bins=9)
plt.title('Distribution of Ratings for the Wolf of Wall Street of Viewers Who Like to Watch Movies Socially')
plt.xlabel('Ratings')
plt.ylabel('Viewers Who Like to Watch Movies Socially')

plt.hist(aloneWWS[:],bins=9)
plt.title('Distribution of Ratings for the Wolf of Wall Street of Viewers Who Like to Watch Movies Alone')
plt.xlabel('Ratings')
plt.ylabel('Viewers Who Like to Watch Movies Alone')

u5, up5 = stats.mannwhitneyu(socialWWS[:],aloneWWS[:]) #Mann-Whitney U test: watch movies socially (with others) and alone
print("- Test statistic U for Mann-Whitney U test of ratings of viewers who prefer watching movies socially and viewers who prefer watching movies alone:", u5)
print("- p-value for Mann-Whitney U test of social viewers and viewers alone ratings:", up5)

#%% Q7:There are ratings on movies from several franchises ([‘Star Wars’, ‘Harry Potter’, ‘The Matrix’, ‘Indiana Jones’, ‘Jurassic Park’, 
#‘Pirates of the Caribbean’, ‘Toy Story’, ‘Batman’]) in this dataset. How many of these are of inconsistent quality, as experienced by 
#viewers?
#star wars:
starWars_movie_list = ['Star Wars: Episode 1 - The Phantom Menace (1999)', 
                       'Star Wars: Episode II - Attack of the Clones (2002)',
                       'Star Wars: Episode IV - A New Hope (1977)',
                       'Star Wars: Episode V - The Empire Strikes Back (1980)',
                       'Star Wars: Episode VI - The Return of the Jedi (1983)',
                       'Star Wars: Episode VII - The Force Awakens (2015)']
starWars = rating[starWars_movie_list]
starWars.dropna(inplace = True)
arrStarWars = starWars.to_numpy()
arrSW1 = arrStarWars[:,0]
arrSW2 = arrStarWars[:,1]
arrSW4 = arrStarWars[:,2]
arrSW5 = arrStarWars[:,3]
arrSW6 = arrStarWars[:,4]
arrSW7 = arrStarWars[:,5]

#first we can look at some descriptive statistics to get an overview of the data
numMovies = 6
descriptivesContainer = np.empty([numMovies,4]) #Initialize as empty
descriptivesContainer[:] = np.NaN  #Filling them with nans to begin with

for ii in range(numMovies):
    descriptivesContainer[ii,0] = np.mean(arrStarWars[:,ii]) # Mean
    descriptivesContainer[ii,1] = np.std(arrStarWars[:,ii]) # SD
    descriptivesContainer[ii,2] = len(arrStarWars[:,ii]) # n
    descriptivesContainer[ii,3] = descriptivesContainer[ii,1]/np.sqrt(descriptivesContainer[ii,2]) # sem

print(descriptivesContainer)

#because the participants are not different people, a dependent (paired) test should be done, so row-wise removal of missing data is implemented


#dependent test for Star Wars 1 and 2
swt1,swp1 = stats.ttest_rel(arrStarWars[:,0],arrStarWars[:,1])
print("p-value for dependent t-test of Star Wars 1 and 2:", swp1)

#dependent test for Star Wars 1 and 4
swt2,swp2 = stats.ttest_rel(arrStarWars[:,0],arrStarWars[:,2])
print("p-value for dependent t-test of Star Wars 1 and 4:", swp2)

#dependent test for Star Wars 1 and 5
swt3,swp3 = stats.ttest_rel(arrStarWars[:,0],arrStarWars[:,3])
print("p-value for dependent t-test of Star Wars 1 and 5:", swp3)

#dependent test for Star Wars 1 and 6
swt4,swp4 = stats.ttest_rel(arrStarWars[:,0],arrStarWars[:,4])
print("p-value for dependent t-test of Star Wars 1 and 6:", swp4)

#dependent test for Star Wars 1 and 7
swt5,swp5 = stats.ttest_rel(arrStarWars[:,0],arrStarWars[:,5])
print("p-value for dependent t-test of Star Wars 1 and 7:", swp5)


swf,swp = stats.f_oneway(arrStarWars[:,0],arrStarWars[:,1],arrStarWars[:,2],arrStarWars[:,3],arrStarWars[:,4],arrStarWars[:,5])
print("- f-statistic for ANOVA of Star Wars: ", swf)
print("- p-value for ANOVA of Star Wars: ", swp)

#harry potter:
harry_movie_list = ['Harry Potter and the Sorcerer\'s Stone (2001)', 
                    'Harry Potter and the Chamber of Secrets (2002)',
                    'Harry Potter and the Goblet of Fire (2005)',
                    'Harry Potter and the Deathly Hallows: Part 2 (2011)']
                       
harry = rating[harry_movie_list]
harry.dropna(inplace = True)
arrHarry = harry.to_numpy()
arrHP1 = arrHarry[:,0]
arrHP2 = arrHarry[:,1]
arrHP3 = arrHarry[:,2]
arrHP4 = arrHarry[:,3]

#first we can look at some descriptive statistics to get an overview of the data
numMovies = 4
descriptivesContainer = np.empty([numMovies,4]) #Initialize as empty
descriptivesContainer[:] = np.NaN  #Filling them with nans to begin with

for ii in range(numMovies):
    descriptivesContainer[ii,0] = np.mean(arrHarry[:,ii]) # Mean
    descriptivesContainer[ii,1] = np.std(arrHarry[:,ii]) # SD
    descriptivesContainer[ii,2] = len(arrHarry[:,ii]) # n
    descriptivesContainer[ii,3] = descriptivesContainer[ii,1]/np.sqrt(descriptivesContainer[ii,2]) # sem

print(descriptivesContainer)

hpf,hpp = stats.f_oneway(arrHarry[:,0],arrHarry[:,1],arrHarry[:,2],arrHarry[:,3])
print("- f-statistic for ANOVA of Harry Potter: ", hpf)
print("- p-value for ANOVA of Harry Potter: ", hpp)


#the matrix:
matrix_list = ['The Matrix (1999)', 
               'The Matrix Reloaded (2003)',
               'The Matrix Revolutions (2003)']
                       
matrix = rating[matrix_list]
matrix.dropna(inplace = True)
arrMatrix = matrix.to_numpy()
arrM1 = arrMatrix[:,0]
arrM2 = arrMatrix[:,1]
arrM3 = arrMatrix[:,2]

#first we can look at some descriptive statistics to get an overview of the data
numMovies = 3
descriptivesContainer = np.empty([numMovies,4]) #Initialize as empty
descriptivesContainer[:] = np.NaN  #Filling them with nans to begin with

for ii in range(numMovies):
    descriptivesContainer[ii,0] = np.mean(arrMatrix[:,ii]) # Mean
    descriptivesContainer[ii,1] = np.std(arrMatrix[:,ii]) # SD
    descriptivesContainer[ii,2] = len(arrMatrix[:,ii]) # n
    descriptivesContainer[ii,3] = descriptivesContainer[ii,1]/np.sqrt(descriptivesContainer[ii,2]) # sem

print(descriptivesContainer)

mf,mp = stats.f_oneway(arrMatrix[:,0],arrMatrix[:,1],arrMatrix[:,2])
print("- f-statistic for ANOVA of Matrix: ", mf)
print("- p-value for ANOVA of Matrix: ", mp)


#Indiana Jones
IJ_list = ['Indiana Jones and the Raiders of the Lost Ark (1981)',
           'Indiana Jones and the Temple of Doom (1984)',
           'Indiana Jones and the Last Crusade (1989)', 
           'Indiana Jones and the Kingdom of the Crystal Skull (2008)']
                       
IJ = rating[IJ_list]
IJ.dropna(inplace = True)
arrIJ = IJ.to_numpy()
arrIJ1 = arrIJ[:,0]
arrIJ2 = arrIJ[:,1]
arrIJ3 = arrIJ[:,2]
arrIJ4 = arrIJ[:,3]

#first we can look at some descriptive statistics to get an overview of the data
numMovies = 4
descriptivesContainer = np.empty([numMovies,4]) #Initialize as empty
descriptivesContainer[:] = np.NaN  #Filling them with nans to begin with

for ii in range(numMovies):
    descriptivesContainer[ii,0] = np.mean(arrIJ[:,ii]) # Mean
    descriptivesContainer[ii,1] = np.std(arrIJ[:,ii]) # SD
    descriptivesContainer[ii,2] = len(arrIJ[:,ii]) # n
    descriptivesContainer[ii,3] = descriptivesContainer[ii,1]/np.sqrt(descriptivesContainer[ii,2]) # sem

print(descriptivesContainer)

ijf,ijp = stats.f_oneway(arrIJ[:,0],arrIJ[:,1],arrIJ[:,2],arrIJ[:,3])
print("- f-statistic for ANOVA of Indiana Jones: ", ijf)
print("- p-value for ANOVA of Indiana Jones: ", ijp)


#Jurassic park
JP_list = ['Jurassic Park (1993)',
           'The Lost World: Jurassic Park (1997)',
           'Jurassic Park III (2001)']
                       
JP = rating[JP_list]
JP.dropna(inplace = True)
arrJP = JP.to_numpy()
arrJP1 = arrJP[:,0]
arrJP2 = arrJP[:,1]
arrJP3 = arrJP[:,2]

#first we can look at some descriptive statistics to get an overview of the data
numMovies = 3
descriptivesContainer = np.empty([numMovies,4]) #Initialize as empty
descriptivesContainer[:] = np.NaN  #Filling them with nans to begin with

for ii in range(numMovies):
    descriptivesContainer[ii,0] = np.mean(arrJP[:,ii]) # Mean
    descriptivesContainer[ii,1] = np.std(arrJP[:,ii]) # SD
    descriptivesContainer[ii,2] = len(arrJP[:,ii]) # n
    descriptivesContainer[ii,3] = descriptivesContainer[ii,1]/np.sqrt(descriptivesContainer[ii,2]) # sem

print(descriptivesContainer)

jpf,jpp = stats.f_oneway(arrJP[:,0],arrJP[:,1],arrJP[:,2])
print("- f-statistic for ANOVA of Jurassic Park: ", jpf)
print("- p-value for ANOVA of Jurassic Park: ", jpp)



#pirates of the caribbean
caribbean_list = ['Pirates of the Caribbean: The Curse of the Black Pearl (2003)',
                  'Pirates of the Caribbean: Dead Man\'s Chest (2006)',
                  'Pirates of the Caribbean: At World\'s End (2007)']
                       
caribbean = rating[caribbean_list]
caribbean.dropna(inplace = True)
arrCaribbean = caribbean.to_numpy()
arrPC1 = arrCaribbean[:,0]
arrPC2 = arrCaribbean[:,1]
arrPC3 = arrCaribbean[:,2]

#first we can look at some descriptive statistics to get an overview of the data
numMovies = 3
descriptivesContainer = np.empty([numMovies,4]) #Initialize as empty
descriptivesContainer[:] = np.NaN  #Filling them with nans to begin with

for ii in range(numMovies):
    descriptivesContainer[ii,0] = np.mean(arrCaribbean[:,ii]) # Mean
    descriptivesContainer[ii,1] = np.std(arrCaribbean[:,ii]) # SD
    descriptivesContainer[ii,2] = len(arrCaribbean[:,ii]) # n
    descriptivesContainer[ii,3] = descriptivesContainer[ii,1]/np.sqrt(descriptivesContainer[ii,2]) # sem

print(descriptivesContainer)

pcf,pcp = stats.f_oneway(arrCaribbean[:,0],arrCaribbean[:,1],arrCaribbean[:,2])
print("- f-statistic for ANOVA of Pirates of the Caribbean: ", pcf)
print("- p-value for ANOVA of Pirates of the Caribbean: ", pcp)



#Toy Story
TS_list = ['Toy Story (1995)',
           'Toy Story 2 (1999)',
           'Toy Story 3 (2010)']
                       
TS = rating[TS_list]
TS.dropna(inplace = True)
arrTS = TS.to_numpy()
arrTS1 = arrTS[:,0]
arrTS2 = arrTS[:,1]
arrTS3 = arrTS[:,2]

#first we can look at some descriptive statistics to get an overview of the data
numMovies = 3
descriptivesContainer = np.empty([numMovies,4]) #Initialize as empty
descriptivesContainer[:] = np.NaN  #Filling them with nans to begin with

for ii in range(numMovies):
    descriptivesContainer[ii,0] = np.mean(arrTS[:,ii]) # Mean
    descriptivesContainer[ii,1] = np.std(arrTS[:,ii]) # SD
    descriptivesContainer[ii,2] = len(arrTS[:,ii]) # n
    descriptivesContainer[ii,3] = descriptivesContainer[ii,1]/np.sqrt(descriptivesContainer[ii,2]) # sem

print(descriptivesContainer)

tsf,tsp = stats.f_oneway(arrTS[:,0],arrTS[:,1],arrTS[:,2])
print("- f-statistic for ANOVA of Toy Story: ", tsf)
print("- p-value for ANOVA of Toy Story: ", tsp)



#Batman
Batman_list = ['Batman (1989)',
               'Batman & Robin (1997)',
               'Batman: The Dark Knight (2008)']
                       
Batman = rating[Batman_list]
Batman.dropna(inplace = True)
arrBatman = Batman.to_numpy()
arrBatman1 = arrBatman[:,0]
arrBatman2 = arrBatman[:,1]
arrBatman3 = arrBatman[:,2]

#first we can look at some descriptive statistics to get an overview of the data
numMovies = 3
descriptivesContainer = np.empty([numMovies,4]) #Initialize as empty
descriptivesContainer[:] = np.NaN  #Filling them with nans to begin with

for ii in range(numMovies):
    descriptivesContainer[ii,0] = np.mean(arrBatman[:,ii]) # Mean
    descriptivesContainer[ii,1] = np.std(arrBatman[:,ii]) # SD
    descriptivesContainer[ii,2] = len(arrBatman[:,ii]) # n
    descriptivesContainer[ii,3] = descriptivesContainer[ii,1]/np.sqrt(descriptivesContainer[ii,2]) # sem

print(descriptivesContainer)

batmanf,batmanp = stats.f_oneway(arrBatman[:,0],arrBatman[:,1],arrBatman[:,2])
print("- f-statistic for ANOVA of Batman: ", batmanf)
print("- p-value for ANOVA of Batman: ", batmanp)


#%% Q8:Build a prediction model of your choice (regression or supervised learning) to predict movie
#ratings (for all 400 movies) from personality factors only. Make sure to use cross-validation
#methods to avoid overfitting and characterize the accuracy of your model.


y_Rating = rating.to_numpy()

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(data.iloc[:, 420:464])
imputed_personality = imputer.transform(data.iloc[:, 420:464].values)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(rating)
imputed_rating = imputer.transform(rating.values)


personality_train, personality_test, rating_train, rating_test = train_test_split(imputed_personality, imputed_rating, test_size = 0.5)

regressor = LinearRegression()
regressor.fit(personality_train, rating_train)

rating_predict = regressor.predict(personality_test)

print(rating_predict)

print("R-squared of the prediction model is:", metrics.r2_score(rating_test, rating_predict))
print("RMSE of the prediction model is:", np.sqrt(metrics.mean_squared_error(rating_test, rating_predict)))

#%% Q9:Build a prediction model of your choice (regression or supervised learning) to predict movie
#ratings (for all 400 movies) from gender identity, sibship status and social viewing preferences (columns 475-477) 
#only. Make sure to use cross-validation methods to avoid overfitting and characterize the accuracy of your model.

#data cleaning (pruning)
#generate an array/ dataframe with 3 columns: gender, sibship, social viewing
#these are our X (predictors)
gender_sibship_social = data.iloc[:, 474:477]

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(gender_sibship_social)
imputed_gss = imputer.transform(gender_sibship_social.values)

gss_train, gss_test, rating_train_gss, rating_test_gss = train_test_split(imputed_gss, imputed_rating, test_size = 0.5)

regressor = LinearRegression()
regressor.fit(gss_train, rating_train_gss)

rating_predict_gss = regressor.predict(gss_test)
print(rating_predict_gss)

print("R-squared of the prediction model is:", metrics.r2_score(rating_test_gss, rating_predict_gss))
print("RMSE of the prediction model is:", np.sqrt(metrics.mean_squared_error(rating_test_gss, rating_predict_gss)))


#%% Q10: Build a prediction model of your choice (regression or supervised learning) to predict movie ratings (for all 400 movies) from all available factors that are not movie ratings (columns 401- 477). Make sure to use cross-validation methods to avoid overfitting and characterize the accuracy of your model.

allFactors = data.iloc[:, 401:477]

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(allFactors)
imputed_af = imputer.transform(allFactors.values)

af_train, af_test, rating_train_af, rating_test_af = train_test_split(imputed_af, imputed_rating, test_size = 0.5)

regressor = LinearRegression()
regressor.fit(af_train, rating_train_af)

rating_predict_af = regressor.predict(af_test)
print(rating_predict_af)

print("R-squared of the prediction model is:", metrics.r2_score(rating_test_af, rating_predict_af))
print("RMSE of the prediction model is:", np.sqrt(metrics.mean_squared_error(rating_test_af, rating_predict_af)))


#%% Extra credit

gender = data.iloc[:, 474]
dfGender = gender.to_frame()
bendList = ['Bend it Like Beckham (2002)']
bendRating = rating[bendList]
bendRatingGender = pd.concat([bendRating, dfGender],axis=1)
bendRatingGender.dropna(inplace = True)

#sort the dataframe according to gender
bendRatingGender.sort_values(by=['Gender identity (1 = female; 2 = male; 3 = self-described)'],inplace=True)
arrBendRatingGender = bendRatingGender.to_numpy()

#put different genders into different arrays
femaleBend= arrBendRatingGender[0:294,0]
maleBend = arrBendRatingGender[294:372,0]


#null hypothesis: the ratings have no difference (all genders rate Shrek equally)
#first take a look at the distributions of ratings for different genders
plt.hist(femaleShrek[:],bins=9)
plt.title('Distribution of Ratings of Female Viewers')
plt.xlabel('Ratings')
plt.ylabel('Female Viewers')
plt.hist(maleShrek[:],bins=9)
plt.title('Distribution of Ratings of Male Viewers')
plt.xlabel('Ratings')
plt.ylabel('Male Viewers')
plt.hist(otherShrek[:],bins=9)
plt.title('Distribution of Ratings of Self-identified-gendered Viewers')
plt.xlabel('Ratings')
plt.ylabel('Self-identified-gendered Viewers')
#clearly, the data isn't normally distributed, so a non-parametric test may be more appropriate
ub,upb = stats.mannwhitneyu(femaleBend[:],maleBend[:]) #Mann-Whitney U test: female and male
print("Test statistic U for Mann-Whitney U test of female and male ratings:", ub)
print("p-value for Mann-Whitney U test of female and male ratings:", upb)
print()

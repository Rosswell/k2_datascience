#  Assignment: Classification
We will be using a data set of pictures (headshots) and answering a few questions. The dataset is called Labeled Faces in the Wild and can be found in http://vis-www.cs.umass.edu/lfw/ or we can download it through sklearn.datasets (this might take a while since it is ~200MB).

The dataset we are about to use is fairly large in feature size thus it requires some manipulation. We will use this opportunity to familiarize ourselves a bit more with PCA and then use PCA for our classification models.

### Exercise 1: Exploring and prepping the data with PCA.
__
Step 1: Open dataset and only select those faces for which we have 70 or more images.

Use the following command to download it:
lfw = datasets.fetch_lfw_people(min_faces_per_person=70, 
                                resize=0.4,
                                data_home='datasets')

__
Step 2: Print a few of the faces to familiarized yourself with the data.

Use the following command: 
fig = plt.figure(figsize=(8, 6))
for i in range(15):
    ax = fig.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(lfw.images[i], cmap=plt.cm.bone)
   
__   
Step 3: Graph the different labels vs their count.

Use the following command:
plt.figure(figsize=(10, 2))
unique_targets = np.unique(lfw.target)
counts = [(lfw.target == i).sum() for i in unique_targets]
plt.xticks(unique_targets, lfw.target_names[unique_targets])
locs, labels = plt.xticks()
plt.setp(labels, rotation=45, size=14)
_ = plt.bar(unique_targets, counts)

__
Step 4: Notice that the number of features in our dataset is fairly large. This is a good moment to apply PCA to reduce the dimensionality of our dataset. Lets choose 150 components.

__
Step 5: A really cool thing about PCA is that it lets you compute the mean of each entry which we can then use to obtain the 'average' face in our dataset.

Run the following command:
plt.imshow(pca.mean_.reshape((50, 37)), cmap=plt.cm.bone)

*Does it look like someone in particular?*

__
Step 6: Plot the components of the PCA. Notice that these are always ordered by importance.

Run the following command:
fig = plt.figure(figsize=(16, 6))
for i in range(30):
    ax = fig.add_subplot(3, 10, i + 1, xticks=[], yticks=[])
    ax.imshow(pca.components_[i].reshape((50, 37)), cmap=plt.cm.bone)

*As you can see the first few components are very good at picking up lighting and the others seem to be identifying things like the nose, eyes or mouth shape.*

### Exercise 2: 
*Part 1*:
Fit the following classification models against our dataset:
1) Logistic Regression
2) KNeighbors Classifier
3) Linear Discriminant
4) Naive Bayes

*Part 2*: Which one had the best performance? Which one had the worst performance?

*Part 3*: Any idea why the score on the top two differs so drastically from the last two?
__
In some cases mean accuracy is not the best measure of performance. Lets evaluate our top model with other methods in the next two problems.

*Part 4*: Find the log_loss, precision, recall, f1_score of the best model.

*Part 5*: Plot the Confusion Matrix of the best model.

*Part 6* (optional):
Edit the code from *Step 2* to display not only the image but also the label and color code the label in red if your model got it wrong or black if it got it right.

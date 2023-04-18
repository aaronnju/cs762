import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from scipy import stats

def outlier_zscore(data, z_scores):
    threshold_list = []
    for threshold in np.arange(1, 5, 0.2):
        threshold_list.append((threshold, len(np.where(z_scores > threshold)[0])))
        df_outlier = pd.DataFrame(threshold_list, columns = ['threshold', 'outlier_count'])
        df_outlier['pct'] = (df_outlier.outlier_count - df_outlier.outlier_count.shift(-1))/df_outlier.outlier_count*100
    plt.plot(df_outlier.threshold, df_outlier.outlier_count)
    best_treshold = round(df_outlier.iloc[df_outlier.pct.argmax(), 0],2)
    outlier_limit = int(np.mean(data) + np.std(data) * df_outlier.iloc[df_outlier.pct.argmax(), 0])
    percentile_threshold = stats.percentileofscore(data, outlier_limit)
    plt.vlines(best_treshold, 0, df_outlier.outlier_count.max(), colors="r", ls = ":")
    plt.annotate("Zscore : {}\nValue : {}\nPercentile : {}".format(best_treshold, outlier_limit,
                                                                   (np.round(percentile_threshold, 3),
                                                                    np.round(100-percentile_threshold, 3))),
                 (best_treshold, df_outlier.outlier_count.max()/2))
    print(df_outlier)

def outlier_inspect(data, z_scores):
    fig = plt.figure(figsize=(20, 6))
    fig.suptitle("age", fontsize=16)
    plt.subplot(1,3,1)
    # Plot univariate or bivariate histograms to show distributions of datasets.
    sns.histplot(data, kde=False, bins = 50, color="r")
    plt.subplot(1,3,2)
    # Draw a box plot to show distributions with respect to categories.
    sns.boxplot(x=data)
    plt.subplot(1,3,3)
    outlier_zscore(data, z_scores)
    plt.show()

def outliesr():
    # age
    data = [13, 15, 16, 16, 19, 20, 20, 21, 22, 22, 25, 25, 25, 25, 30, 33, 33, 35, 35, 35, 35, 36, 40, 45, 46, 52, 70]
    # Calculate mean and standard deviation
    mean = np.mean(data)
    std = np.std(data)
    # Calculate z-scores for each data point
    z_scores = [(x - mean) / std for x in data]

    outlier_inspect(data, z_scores)

    # Identify any outliers, using threshold = 3
    outliers = [data[i] for i in range(len(data)) if z_scores[i] < -3 or z_scores[i] > 3]
    print(outliers)


# Define the data
data = np.array(
    [13, 15, 16, 16, 19, 20, 20, 21, 22, 22, 25, 25, 25, 25, 30, 33, 33, 35, 35, 35, 35, 36, 40, 45, 46, 52, 70])

# Define the number of bins
num_bins = 3

# Calculate the bin boundaries
bin_boundaries = np.linspace(data.min(), data.max(), num_bins + 1)

# Calculate the bin centers
bin_centers = 0.5 * (bin_boundaries[1:] + bin_boundaries[:-1])

# Calculate the bin index for each data point
bin_index = np.digitize(data, bin_boundaries[1:-1])

# Calculate the mean of each bin
bin_means = np.array([data[bin_index == i].mean() for i in range(1, num_bins + 1)])

# Print the bin boundaries, bin centers, and bin means
print('Bin boundaries:', bin_boundaries)
print('Bin centers:', bin_centers)
print('Bin means:', bin_means)

# Plot the original data as a scatter plot
plt.scatter(range(len(data)), data, label='Original data')

# Plot the smoothed data as a line graph
plt.plot(bin_centers, bin_means, '-o', label='Smoothed data')

# Add labels and legend
plt.xlabel('Index')
plt.ylabel('Age')
plt.title('Smoothing by bin means')
plt.legend()

# Show the plot
plt.show()

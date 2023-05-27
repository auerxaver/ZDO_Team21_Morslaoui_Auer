import os
import numpy as np
import json
import statistics as stats
import matplotlib.pyplot as plt



class Analysis:
    def __init__(self, output_filename):
        self.dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.distances = []
        self.dist_scores = []
        self.angle_scores = []
        with open(os.path.join(self.dir_path, "out", output_filename)) as f:
            self.meta = json.load(f)
        for idx in range(len(self.meta)):
            self.stitch_distance(idx)
            self.stitch_angle(idx)

        self.visualization()

    # Calculate score based on similarity of distances between stitches
    def stitch_distance(self, idx):
        distances = []
        cr_positions = self.meta[idx]['crossing_positions']
        for i in range(len(cr_positions) - 1):
            distances.append(abs(cr_positions[i+1] - cr_positions[i]))
        self.calc_dist_score(distances)
        self.distances.append(distances)

    # Calculate score based on perpendicularity of stitches in regards to the incision
    def stitch_angle(self, idx):
        perfect_angle = 90
        angles_from_perfect = []
        for angle in self.meta[idx]['crossing_angles']:
            angles_from_perfect.append(abs(angle) - perfect_angle)
        self.calc_angle_score(angles_from_perfect)

    def calc_dist_score(self, distances):
        if len(distances) > 1:
            ell = distances
            mode_measure = 1 - len(set(ell))/len(ell)
            avg_measure = 1 - stats.stdev(ell)/stats.mean(ell)
            self.dist_scores.append(max(avg_measure, mode_measure)**2)
        else:
            self.dist_scores.append(0)

    def calc_angle_score(self, angles_from_perfect):
        average_deviation = abs(np.mean(angles_from_perfect))
        score = (1.0 - average_deviation/100)**4
        self.angle_scores.append(score)


    def visualization(self):
        images = [self.meta[idx]['filename'] for idx in range(len(self.meta))]
        penguin_means = {
            'Distance Score': self.dist_scores,
            'Angle Score': self.angle_scores,
        }
        x = np.arange(len(self.meta))  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0

        fig, ax = plt.subplots(layout='constrained')

        for attribute, measurement in penguin_means.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute)
            ax.bar_label(rects, padding=3)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Score (0-1)')
        ax.set_title('Results')
        ax.set_xticks(x + width, images)
        ax.legend(loc='upper left', ncols=3)
        ax.set_ylim(0, 1.2)

        plt.show()
        test = 1
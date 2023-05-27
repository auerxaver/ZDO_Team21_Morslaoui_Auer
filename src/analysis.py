import os
import json
import statistics as stats



class Analysis:
    def __init__(self, output_filename):
        self.dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.distances = []
        self.dist_scores = []
        self.angle_from_perfect = []
        with open(os.path.join(self.dir_path, "out", output_filename)) as f:
            self.meta = json.load(f)
        for idx in range(len(self.meta)):
            self.stitch_distance(idx)
            self.stitch_angle()

        test = 0

    # Calculate score based on similarity of distances between stitches
    def stitch_distance(self, idx):
        distances = []
        cr_positions = self.meta[idx]['crossing_positions']
        for i in range(len(cr_positions) - 1):
            distances.append(abs(cr_positions[i+1] - cr_positions[i]))
        self.calc_dist_score(distances)
        self.distances.append(distances)

    # Calculate score based on perpendicularity of stitches in regards to the incision
    def stitch_angle(self):
        return

    def calc_dist_score(self, distances):
        if len(distances) > 1:
            ell = distances
            mode_measure = 1 - len(set(ell))/len(ell)
            avg_measure = 1 - stats.stdev(ell)/stats.mean(ell)
            self.dist_scores.append(max(avg_measure, mode_measure))
        else:
            self.dist_scores.append('not enough stitches found')

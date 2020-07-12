import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


data = {
    'RHD': [0.6564078157637292, 0.7067105576267153, 0.7490415377700219, 0.783810581423019, 0.813027154226488, 0.836995320060284, 0.8577245445651887, 0.8741836545834325, 0.8881441527193887, 0.8998175616720869, 0.9100896327437138, 0.9187752835726184, 0.9261389175325877, 0.9324317178287195, 0.9379180878348008, 0.9425451468760742],
    'STB': [0.7323492063492063, 0.7761904761904762, 0.8114920634920635, 0.841015873015873, 0.8640952380952381, 0.8826825396825397, 0.8992380952380953, 0.9118888888888889, 0.9234761904761905, 0.9322698412698412, 0.9411428571428572, 0.9485873015873015, 0.9547936507936507, 0.9609047619047619, 0.9655238095238096, 0.9693015873015873],
    'DO': [0.8161910144108505, 0.8587877931619101, 0.8924837524724498, 0.9169256852218141, 0.9356456626165583, 0.9489968917773383, 0.9583922011867759, 0.9655269850240181, 0.9708957332579825, 0.9742158801921447, 0.977606668550438, 0.9791607798813224, 0.9812800226052557, 0.9829047753602712, 0.9837524724498445, 0.9843176038428935],
    'ED': [0.6108895439569967, 0.6606554534419976, 0.7024449453788798, 0.7398994277787411, 0.7724986994971389, 0.7964279521414948, 0.8172359979191954, 0.8319750303450668, 0.8449800589561297, 0.8545170799375759, 0.864054100919022, 0.873764522281949, 0.8801803363967401, 0.8838217444078377, 0.8883301543263395, 0.8923183631003988]
}


def calculate_auc(xs, ys):
    length = xs[-1] - xs[0]
    area = 0
    for i in range(len(ys) - 1):
        area += (ys[i] + ys[i + 1]) * (xs[i + 1] - xs[i]) / 2 / length
    return area


def plot_pck():
    for name, ys in data.items():
        xs = np.linspace(20, 50, len(ys))
        label = name + (' (AUC=%.3f)' % calculate_auc(xs, ys))
        plt.plot(xs, ys, label=label)
    plt.legend()
    plt.xlabel('Error thresholds (mm)')
    plt.ylabel('3D PCK')
    plt.grid(linestyle='-.')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_pck()
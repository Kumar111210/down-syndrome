import numpy as np

def analyze_heatmap_regions(heatmap):
    h, w = heatmap.shape
    regions = {}
    regions["forehead"] = np.mean(heatmap[0:int(h*0.2), :])
    regions["eyes"] = np.mean(heatmap[int(h*0.2):int(h*0.4), :])
    regions["nose"] = np.mean(heatmap[int(h*0.4):int(h*0.6), :])
    regions["mouth"] = np.mean(heatmap[int(h*0.6):int(h*0.8), :])
    regions["chin"] = np.mean(heatmap[int(h*0.8):, :])

    threshold = 0.2
    affected = [r for r,v in regions.items() if v>threshold]
    if not affected:
        affected = ["general face"]
    return affected

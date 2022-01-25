# VNet keras
#
# Joo-won Kim

import math

def _dice(dat1, dat2):
    """calculate dice similarity coefficient of two segmentation
    2 * |A and B| / (|A| + |B|)"""
    total = len(dat1.nonzero()[0]) + len(dat2.nonzero()[0])
    intersect = np.logical_and(dat1, dat2)

    return 2.0*len(intersect.nonzero()[0])/total


def dice(dat1, dat2):
    """calculate dice similarity coefficient of two segmentation
    2 * |A and B| / (|A| + |B|)"""
    return 2.0*np.logical_and(dat1, dat2).sum() / (dat1.sum() + dat2.sum())


def dice_prob(dat1, dat2):
    """calculate dice similarity coefficient of two probability maps
    2 * |min(A, B)| / (|A| + |B|)"""
    total = dat1.sum() + dat2.sum()
    intersect = np.zeros(dat1.shape, dtype=dat1.dtype)
    intersect[dat1<=dat2] = dat1[dat1<=dat2]
    intersect[dat1>dat2] = dat2[dat1>dat2]

    return 2.0*intersect.sum()/total


def hausdorff_distance(dat1, dat2, zooms=None, init_distance=1000000.0):
    """calculate Hausdorff distance, the maximum of surface distances of two segmentation
    max( d(a, B) | a in A, d(b, A) | b in B ), where d(x, Y):= min{ d(x, y) | y in Y} )"""
    if zooms is not None:
        do_real_unit = True
        dx, dy, dz = zooms
        dxx = dx*dx
        dyy = dy*dy
        dzz = dz*dz
    else:
        do_real_unit = False

    tmp = dat1.nonzero()
    seg1 = [(tmp[0][i], tmp[1][i], tmp[2][i]) for i in range(len(tmp[0]))]
    tmp = dat2.nonzero()
    seg2 = [(tmp[0][i], tmp[1][i], tmp[2][i]) for i in range(len(tmp[0]))]

    max_distance_voxel = 0.0
    if do_real_unit:
        max_distance = 0.0
    for da, db, sega, segb in ( (dat1, dat2, seg1, seg2), (dat2, dat1, seg2, seg1) ):
        for (i,j,k) in sega:
            if db[i,j,k] > 0:
                continue
            if da[i+1,j,k] > 0 and da[i-1,j,k] > 0 and da[i,j+1,k] > 0 and da[i,j-1,k] > 0 and da[i,j,k+1] > 0 and da[i,j,k-1] > 0:
                continue
            min_distance_voxel = init_distance
            if do_real_unit:
                min_distance = init_distance
            for (ib, jb, kb) in segb:
                distance_voxel = (i-ib)**2 + (j-jb)**2 + (k-kb)**2
                if distance_voxel < min_distance_voxel:
                    min_distance_voxel = distance_voxel
                if do_real_unit:
                    distance = (i-ib)**2*dxx + (j-jb)**2*dyy + (k-kb)**2*dzz
                    if distance < min_distance:
                        min_distance = distance
            if max_distance_voxel < min_distance_voxel:
                max_distance_voxel = min_distance_voxel
            if do_real_unit and (max_distance < min_distance):
                max_distance = min_distance

    if do_real_unit:
        return math.sqrt(max_distance_voxel), math.sqrt(max_distance)
    else:
        return math.sqrt(max_distance_voxel)



def mean_distance(dat1, dat2, zooms=None, init_distance=10000.0):
    """calculate mean distance, the mean of surface distances of two segmentation
    mean( d(a, B) | a in A , d(b, A) | b in B ), where d(x, Y):= min{ d(x, y) | y in Y} )"""
    if zooms is not None:
        do_real_unit = True
        dx, dy, dz = zooms
        dxx = dx*dx
        dyy = dy*dy
        dzz = dz*dz
    else:
        do_real_unit = False

    seg1 = []
    seg2 = []
    for da, seg in ((dat1, seg1), (dat2, seg2)):
        tmp = da.nonzero()
        for (i,j,k) in zip(tmp[0], tmp[1], tmp[2]):
            if da[i+1,j,k] > 0 and da[i-1,j,k] > 0 and da[i,j+1,k] > 0 and da[i,j-1,k] > 0 and da[i,j,k+1] > 0 and da[i,j,k-1] > 0:
                continue
            seg.append((i,j,k))

    sum_distance_voxel = 0.0
    if do_real_unit:
        sum_distance = 0.0
    count = 0
    for sega, segb in ((seg1, seg2), (seg2, seg1)):
        for (i,j,k) in sega:
            count += 1
            if (i,j,k) in segb:
                continue
            min_distance_voxel = init_distance
            if do_real_unit:
                min_distance = init_distance
            for (ib, jb, kb) in segb:
                distance_voxel = (i-ib)**2 + (j-jb)**2 + (k-kb)**2
                if distance_voxel < min_distance_voxel:
                    min_distance_voxel = distance_voxel
                if do_real_unit:
                    distance = (i-ib)**2*dxx + (j-jb)**2*dyy + (k-kb)**2*dzz
                    if distance < min_distance:
                        min_distance = distance
            sum_distance_voxel += math.sqrt(min_distance_voxel)
            if do_real_unit:
                sum_distance += math.sqrt(min_distance)

    if do_real_unit:
        return sum_distance_voxel/count, sum_distance/count
    else:
        return sum_distance_voxel/count


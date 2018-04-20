'''GC map definition:
# 1. background.
# 2. frontal wall
# 3. left wall
# 4. right wall
# 5. floor
# 6. ceiling'''
'''Output definition:
linelabels: (9 in total)
'23': between frontal wall (2) and left wall (3)
'24': between frontal wall (2) and right wall (4)
'25': between frontal wall (2) and floor (5)
'26': between frontal wall (2) and ceiling (6)
'35': between left wall (2) and floor (5)
'36': between left wall (2) and ceiling (6)
'45': between right wall (2) and floor (5)
'46': between right wall (2) and ceiling (6)'''

gc_def = {}

gc_def['backgroundID'] = 1
gc_def['frontal_wallID'] = 2
gc_def['left_wallID'] = 3
gc_def['right_wallID'] = 4
gc_def['floor_ID'] = 5
gc_def['ceiling_ID'] = 6

gc_neighbours = [[gc_def['frontal_wallID'], gc_def['left_wallID']],
                 [gc_def['frontal_wallID'], gc_def['right_wallID']],
                 [gc_def['frontal_wallID'], gc_def['floor_ID']],
                 [gc_def['frontal_wallID'], gc_def['ceiling_ID']],
                 [gc_def['left_wallID'], gc_def['floor_ID']],
                 [gc_def['left_wallID'], gc_def['ceiling_ID']],
                 [gc_def['right_wallID'], gc_def['floor_ID']],
                 [gc_def['right_wallID'], gc_def['ceiling_ID']]]

neighbour_clusters = [[[gc_def['frontal_wallID'], gc_def['left_wallID']],
                       [gc_def['frontal_wallID'], gc_def['right_wallID']]],
                      [[gc_def['frontal_wallID'], gc_def['floor_ID']],
                       [gc_def['frontal_wallID'], gc_def['ceiling_ID']]],
                      [[gc_def['left_wallID'], gc_def['floor_ID']],
                       [gc_def['left_wallID'], gc_def['ceiling_ID'] ],
                       [gc_def['right_wallID'], gc_def['floor_ID']],
                       [gc_def['right_wallID'], gc_def['ceiling_ID']]]]

neighbour_cluster_set = []

for cluster in neighbour_clusters:
    cluster_set = []
    for plabel1, plabel2 in cluster:
        if plabel1 < plabel2:
            label = plabel1 * 10 + plabel2
        else:
            label = plabel2 * 10 + plabel1
        cluster_set.append(label)
    neighbour_cluster_set.append(cluster_set)


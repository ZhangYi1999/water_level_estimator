for kpt,kpt_id in zip(car.original_keypoints,car.keypoints_id):
            plt.scatter(int(kpt[0]),int(kpt[1]),c='coral', s=10)
            plt.text(int(kpt[0]), int(kpt[1]), str(kpt_id),color = "ivory",fontsize=10)
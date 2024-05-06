

def contour_process(x_points, y_points, hierarchy, *args, **kwargs):
    # print(x_points, y_points, hierarchy, args, kwargs)
    if len(x_points) == 0 or len(y_points) == 0: return x_points, y_points  # check if contour is empty
    
    if ("external_noise_size" in kwargs) and hierarchy == -1:
        external_noise_size = kwargs["external_noise_size"]
        x_points, y_points = ctr_external_denoise(x_points, y_points, noise_size=external_noise_size)
        if len(x_points) == 0 or len(y_points) == 0: return x_points, y_points

    if ("internal_noise_size" in kwargs) and hierarchy != -1:
        internal_noise_size = kwargs["internal_noise_size"]
        x_points, y_points = ctr_internal_denoise(x_points, y_points, noise_size=internal_noise_size)
        if len(x_points) == 0 or len(y_points) == 0: return x_points, y_points

    return x_points, y_points



def ctr_internal_denoise(x_points, y_points, noise_size):
    if len(x_points) < noise_size or len(y_points) < noise_size:
        x_points = []
        y_points = []
    return x_points, y_points


def ctr_external_denoise(x_points, y_points, noise_size):
    if len(x_points) < noise_size or len(y_points) < noise_size:
        x_points = []
        y_points = []
    return x_points, y_points



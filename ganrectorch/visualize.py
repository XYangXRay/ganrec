import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io

def shorten(string):
    if 'e' in string:
        left = string.split('e')[0][:7]
        right = string.split('e')[1][:7]
        return left + 'e' + right
    else:
        if '.' in string:
            count = 0
            for i in range(len(string.split('.')[1])):
                if string[i] == '0':
                    count += 1
            return string[:count+5]
        else:
            return string[:7]
        
def give_title(image, title = '', idx = '', min_max = True):    
    if min_max:
        min_val_orig = np.min(image)
        max_val_orig = np.max(image)
        txt_min_val = shorten(str(min_val_orig))
        txt_max_val = shorten(str(max_val_orig))
    else:
        txt_min_val = ''
        txt_max_val = ''
    title = 'im='+ str(idx+1) if title == '' else title
    return title + ' (' + txt_min_val + ', ' + txt_max_val + ')'

def give_titles(images, titles = [], min_max = True):
    titles = [titles] if type(titles) is not list else titles
    if len(titles) <= len(images):
        titles = [give_title(images[i], title = titles[i], idx=i, min_max = min_max) for i in range(len(titles))]
        n_for_rest = np.arange(len(titles), len(images))
        titles.extend([give_title(images[i], idx=i, min_max = min_max) for i in n_for_rest])
    else:
        titles = [give_title(images[i], title = titles[i], idx=i, min_max = min_max) for i in range(len(images))]
    return titles

def get_row_col(images, show_all = False):
    if show_all:
        rows = int(np.sqrt(len(images)))
        cols = int(np.sqrt(len(images)))
        return rows, cols + (len(images) - rows*cols)//rows
    
    if len(images) == 1:
        rows = 1
        cols = 1
    elif len(images) <= 5:
        rows = 1
        cols = len(images)
    else:
        rows = 2
        cols = len(images)//2
    if rows*cols > len(images):
        images = images[:rows*cols - int(rows*cols/len(images))]
        print('rows: ', rows, 'cols: ', cols, 'images: ', len(images))
        rows, cols = get_row_col(images)
    print('rows: ', rows, 'cols: ', cols)
    return rows, cols

def chose_fig(images, idx = None, rows = None, cols = None, show_all = False, add_length = None):
    (rows, cols) = get_row_col(images, show_all=show_all) if rows is None or cols is None else (rows, cols)
    shape = images[0].shape
    if shape[0] > 260:
        fig_size = (shape[1]*cols/100+1, shape[0]*rows/100)
    elif shape[0] > 100 and shape[0] <= 260:
        fig_size = (shape[1]*cols/50+1, shape[0]*rows/50)
    else:
        fig_size = (shape[1]*cols/25+1, shape[0]*rows/25)
    if add_length is not None:
        fig_size = (fig_size[0]+add_length, fig_size[1])
    fig, ax = plt.subplots(rows, cols, figsize=fig_size, squeeze=False)
    ax.reshape(rows, cols)
    if rows == 1 and cols == 1:
        return fig, ax, rows, cols, fig_size
    elif rows == 1:
        ax = ax.reshape(1, cols)
        return fig, ax, rows, cols, fig_size
    elif cols == 1:
        ax = ax.reshape(rows, 1)
        return fig, ax, rows, cols, fig_size
    else:
        return fig, ax, rows, cols, fig_size
    
def get_setup_info(dict = {}):
    #rearrange them in a descending order based on length
    dict = {k: v for k, v in sorted(dict.items(), key=lambda item: len(item[0]) + len(str(item[1])), reverse=True)}
    len_line = 0
    for key, value in dict.items():
        if type(value) == str or  type(value) == int or type(value) == float or type(value) == bool: 
            if len(key) > len_line:
                len_line = len(key)
        elif type(value) == np.ndarray:
            if len(value.shape) == 0:
                if len(key) > len_line:
                    len_line = len(key)
        else: 
            try:
                from ganrec_dataloader import tensor_to_np
                if type(tensor_to_np(value)) == np.ndarray and len(tensor_to_np(value).shape) == 0:
                    if len(key) > len_line:
                        len_line = len(key)
            except:
                pass
    len_line += 10
    line = '_'*len_line 
    information = line + '\n'
    for key, value in dict.items():
        if type(value) == str or type(value) == int or type(value) == float or type(value) == bool:
            information += '| ' +key +': '+ str(value) +' \n'
        elif type(value) == np.ndarray and len(value.shape) == 0:
            information += '| ' +key +': '+ str(value) +' \n'
        else:
            try:
                from ganrec_dataloader import tensor_to_np
                if type(tensor_to_np(value)) == np.ndarray and len(tensor_to_np(value).shape) == 0:
                    information += '| ' +key +': '+ str(tensor_to_np(value)) +' \n'
            except:
                pass
    information += line + ' \n'
    return information, len_line

def get_file_nem(dict):
    name = ''
    important_keys = ['experiment_name', 'abs_ratio', 'iter_num', 'downsampling_factor', 'l1_ratio', 'contrast_ratio', 'normalized_ratio', 'brightness_ratio', 'contrast_normalize_ratio', 'brightness_normalize_ratio', 'l2_ratio', 'fourier_ratio']
    for key in important_keys:
        if key in dict.keys():
            name += key + '_' + str(dict[key]) + '__'
    return name
  
def create_table_info(dict={}):
    import pandas as pd
    df = pd.DataFrame()
    for key, value in dict.items():
        if type(value) != np.ndarray:
            df[key] = [value]
        elif type(value) == np.ndarray and len(value.shape) == 0:
            df[key] = [value]
    df = df.T
    #create a plot with the information
    fig, ax = plt.subplots(figsize=(20, 10))
    #make the rows and columns look like a table
    ax.axis('tight')
    ax.axis('off')
    #create the table
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', rowLabels=df.index, cellLoc='center')
    #change the font size
    table.set_fontsize(14)
    #change the cell height
    table.scale(1, 2)
    
    return df,ax, table

def give_titles(images, titles = [], min_max = True):
    titles = [titles] if type(titles) is not list else titles
    if len(titles) <= len(images):
        titles = [give_title(images[i], title = titles[i], idx=i, min_max = min_max) for i in range(len(titles))]
        n_for_rest = np.arange(len(titles), len(images))
        titles.extend([give_title(images[i], idx=i, min_max = min_max) for i in n_for_rest])
    else:
        titles = [give_title(images[i], title = titles[i], idx=i, min_max = min_max) for i in range(len(images))]
    return titles
                       
def val_from_images(image, type_of_image = 'nd.array'):
    if 'ndarray' in str(type_of_image):
        if len(image.shape) == 2:
            val = image
        elif len(image.shape) == 3:
            val = [image[j,:,:] for j in range(len(image))]
        else:
            val = [image[j,0,:,:] for j in range(len(image))]
    elif 'Tensor' in str(type_of_image):
        from ganrec_dataloader import tensor_to_np
        image = tensor_to_np(image)
        if type(image) is not list:
            if len(image.shape) == 2:
                val = image
            elif len(image.shape) == 3:
                val = [image[j,:,:] for j in range(len(image))]
            elif len(image.shape) == 4:
                val = [image[j,0,:,:] for j in range(len(image))]
            elif len(image.shape) == 1:
                val = image
        else:
            val = image
    elif 'jax' in str(type_of_image):
        #jax to numpy
        image = np.array(image)
        if len(image.shape) == 2:
            val = image
        elif len(image.shape) == 3:
            val = [image[j,:,:] for j in range(len(image))]
        elif len(image.shape) == 4:
            val = [image[j,0,:,:] for j in range(len(image))]
        elif len(image.shape) == 1:
            val = image
        else:
            val = image
    elif type_of_image == 'str':
        val = io.imread_collection(image)
    elif 'collection' in str(type_of_image):
        val = image
    elif 'list' in str(type_of_image):
        val = [val_from_images(image, type_of_image = type(image)) for image in image]
    else:
        assert False, "type_of_image is not nd.array, list or torch.Tensor"
    return val
    
def convert_images(images, idx = None):
    if idx is not None:
        images = [images[i] for i in idx]
    if type(images) is list:
        vals = [val_from_images(image, type_of_image = type(image)) for image in images]
  
        for i, val in enumerate(vals):
            if type(val) is list:
                [vals.append(val[j]) for j in range(len(val))]
                vals.pop(i)
        images = vals
    else:
        images = val_from_images(images, type_of_image = type(images))
    for i, val in enumerate(images):
        if type(val) is list:
            [images.append(val[j]) for j in range(len(val))]
            images.pop(i)
    return images

def visualize(images, idx = None, rows = None, cols = None, show_or_plot = 'show', cmap = 'coolwarm', title = '', axis = 'on', plot_axis = 'half', min_max = True, dict = None, save_path=None, save_name=None, show_all = False):
    """
    Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r',
    """
    #'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r',
    images = convert_images(images, idx)
    titles = give_titles(images, title, min_max)
    shape = images[0].shape
    
    if dict is not None:
        description_title, add_length = get_setup_info(dict)
    else:
        add_length = None
    fig, ax, rows, cols, fig_size= chose_fig(images, idx, rows, cols, add_length, show_all)
   
    if show_or_plot == 'plot':
        if plot_axis == 'half':
            [ax[i, j].plot(images[i*cols + j][shape[0]//2, :]) for i in range(rows) for j in range(cols)]
        else:
            assert type(plot_axis) == int, "plot_axis is not 'half' or an integer"
            [ax[i, j].plot(images[i*cols + j][plot_axis, :]) for i in range(rows) for j in range(cols)]
    elif show_or_plot == 'both':
        [ax[i, j].imshow(images[i*cols + j], cmap = cmap) for i in range(rows) for j in range(cols)]
        if plot_axis == 'half':
            [ax[i, j].twinx().plot(images[i*cols + j][shape[0]//2, :]) for i in range(rows) for j in range(cols)]
        else:
            assert type(plot_axis) == int, "plot_axis is not 'half' or an integer"
            [ax[i, j].twinx().plot(images[i*cols + j][plot_axis, :]) for i in range(rows) for j in range(cols)]
    
    [ax[i, j].axis(axis) for i in range(rows) for j in range(cols)]
    [ax[i, j].set_title(titles[i*cols + j], fontsize = 12) for i in range(rows) for j in range(cols)]
    plt.tight_layout()
    if show_or_plot != 'plot':
        [fig.colorbar(ax[i, j].imshow(images[i*cols + j], cmap = cmap), ax=ax[i, j]) for i in range(rows) for j in range(cols)]
    fig.patch.set_facecolor('xkcd:light grey')
    
    if dict is not None:
        fig.subplots_adjust(left=add_length/150)
        fig.suptitle(description_title, fontsize=10, y=0.95, x=0.05, ha='left', va='center', wrap=True, color='blue')
    plt.show()
    if save_path is not None:
        if save_name is None:
            save_name = get_file_nem(dict)
        save_path = save_path + save_name + '.png'
        plt.savefig(save_path)

def sns_visualize(images, idx = None, rows = None, cols = None, show_or_plot = 'show', cmap = 'coolwarm', title = '', axis = 'off', plot_axis = 'half'):
    """
    Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r',
    """
    
    import seaborn as sns
    images = convert_images(images, idx)
    titles = give_titles(images, title)
    shape = images[0].shape
    fig, ax, rows, cols = chose_fig(images, idx, rows, cols)

    if rows == 1 and cols == 1:
        if show_or_plot == 'plot':
            if plot_axis == 'half':
                ax.plot(images[0][shape[0]//2, :])
            else:
                assert type(plot_axis) == int, "plot_axis is not 'half' or an integer"
                ax.plot(images[0][plot_axis, :])
            ax.set_title('y:'+str(plot_axis)+' '+titles[0], fontsize = 12)
        elif show_or_plot == 'both':
            if plot_axis == 'half':
                ax.twinx().plot(images[0][shape[0]//2, :])
            else:
                assert type(plot_axis) == int, "plot_axis is not 'half' or an integer"
                ax.twinx().plot(images[0][plot_axis, :])
            ax.set_title('y:'+str(plot_axis)+' '+titles[0], fontsize = 12)
        else:
            ax.set_title(titles[0], fontsize = 12)
            ax.imshow(images[0], cmap = cmap)
        
        ax.axis(axis)
        fig.colorbar(ax.imshow(images[0]), ax=ax)
        plt.show()
        return fig
    if show_or_plot == 'show':    
        [sns.heatmap(images[i*cols + j], cmap = cmap, ax = ax[i, j], robust=True) for i in range(rows) for j in range(cols)]
        [ax[i, j].set_title(titles[i*cols + j], fontsize = 12) for i in range(rows) for j in range(cols)]
    elif show_or_plot == 'plot':
        if plot_axis == 'half':
            [ax[i, j].plot(images[i*cols + j][shape[0]//2, :]) for i in range(rows) for j in range(cols)]
        else:
            assert type(plot_axis) == int, "plot_axis is not 'half' or an integer"
            [ax[i, j].plot(images[i*cols + j][plot_axis, :]) for i in range(rows) for j in range(cols)]
        [ax[i, j].set_title('y:'+str(plot_axis)+' '+titles[i*cols + j], fontsize = 12) for i in range(rows) for j in range(cols)]
    elif show_or_plot == 'both':
        [sns.heatmap(images[i*cols + j], cmap = cmap, ax = ax[i, j]) for i in range(rows) for j in range(cols)]
        if plot_axis == 'half':
            [ax[i, j].twinx().plot(images[i*cols + j][shape[0]//2, :]) for i in range(rows) for j in range(cols)]
        else:
            assert type(plot_axis) == int, "plot_axis is not 'half' or an integer"
            [ax[i, j].twinx().plot(images[i*cols + j][plot_axis, :]) for i in range(rows) for j in range(cols)]
        [ax[i,j].set_title('y:'+str(plot_axis)+' '+titles[i*cols + j], fontsize = 12) for i in range(rows) for j in range(cols) ]
    else:
        assert False, "show_or_plot is not 'show', 'plot' or 'both'"
    [ax[i, j].axis(axis) for i in range(rows) for j in range(cols)]
    plt.tight_layout()
    fig.patch.set_facecolor('xkcd:light blue')
    plt.show()
    return fig

def visualize_interact(pure = []):
    import ipywidgets as widgets
    from ipywidgets import interact
    from IPython.display import display
    interact(visualize, pure = widgets.fixed(pure), show_or_plot = widgets.Dropdown(options=['show', 'plot'], value='show', description='Show or plot:'), rows = widgets.IntSlider(min=1, max=10, step=1, value=1, description='Rows:'), cols = widgets.IntSlider(min=1, max=10, step=1, value=3, description='Columns:'))
 
def plot_pandas(df, column_range = None, x_column = None, titles = None):
    """
    this function plots the metadata dataframe
    """
    if column_range is None:
        column_range = df.columns[2:-1]
    elif column_range == 'all':
        column_range = df.columns
    elif type(column_range) is str:
        column_range = [column_range]
    elif type(column_range) is int:
        column_range = df.columns[column_range:-1]

    rows, cols = get_row_col(column_range)
    print('rows: ', rows, 'cols: ', cols)
    if rows*cols < 6:  
        fig = plt.figure(figsize=(10,5))
    else:
        fig = plt.figure(figsize=(20,10))
    min_vals = [df[column].min() for column in column_range], [df[column].idxmin() for column in column_range]
    max_vals = [df[column].max() for column in column_range], [df[column].idxmax() for column in column_range]
    if titles is None:
        # titles = [column + '\nmin = ' + str(min_per_column[i])+' at ' + str(df[column].idxmin()) +'\n max = ' + str(df[column].max())+' at ' + str(df[column].idxmax()) for i, column in enumerate(column_range)]
        titles = [column + '\nmin = ' + str(min_vals[0][i])+' at ' + str(min_vals[1][i]) +'\n max = ' + str(max_vals[0][i])+' at ' + str(max_vals[1][i]) for i, column in enumerate(column_range)]

    

    for i, column in enumerate(column_range):
        ax = fig.add_subplot(rows, cols, i+1)
        if x_column is None:
            ax.plot(df[column])
            ax.set_xlabel('iterations')
            ax.set_ylabel(column)
            ax.set_title(titles[i])
        else:
            ax.plot(df[x_column], df[column])
            ax.set_xlabel(x_column)
            ax.set_ylabel(column)
            ax.set_title(titles[i])
    plt.tight_layout()
    return min_vals, max_vals

def plot_image(plots, idx = None, title = '', fig = None, ax = None):
    if type(plots) is not list:
            plots = [plots]
    if idx is not None:
        plots = [plots[i] for i in idx]
    title = give_titles(plots, title)
    fig_size = (5,10) if len(plots) > 1 else (5,5)
    fig = plt.figure(figsize=fig_size) if fig is None else fig
    ax = fig.add_subplot(111)
    [ax.plot(plots[i]) for i in range(len(plots))]
    ax.set_title(title)
    ax.legend(title)
    plt.show()
    return fig, ax

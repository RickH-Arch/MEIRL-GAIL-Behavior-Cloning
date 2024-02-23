{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[[2], [4]]"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "d = {1:[2],3:[4]}\n",
    "list(d.values())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "l = np.load(r'wifi_track_data\\dacang\\train_data\\rewards_deep_irl_fc_epoch94_0115.npy')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([  1,   4,   9,  16,  25,  36,  49,  64,  81, 100])"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "n = np.array([1,2,3,4,5,6,7,8,9,10])\n",
    "n = n*n\n",
    "n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Processing:  95%|█████████▌| 2.0000000000000004/2.1 [00:02<00:00,  1.09s/it] \n"
     ]
    }
   ],
   "source": [
    "import time\n",
    "from tqdm import tqdm\n",
    "\n",
    "with tqdm(total=2.1) as pbar:\n",
    "    pbar.set_description('Processing')\n",
    "    # 手动设置：每次更新10个进度，一共更新20次，总共更新total：200\n",
    "    for i in range(20):\n",
    "         # 进行动作, 这里是过0.1s\n",
    "         time.sleep(0.1)\n",
    "         # 进行进度更新, 这里设置10个\n",
    "         pbar.update(0.1)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "float"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "delta = np.inf\n",
    "delta = 0.4\n",
    "thre = 0.01\n",
    "type(delta-thre)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 1, 1, 2, 2])"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "l = np.array([-1,1,1,2,-2])\n",
    "l.__abs__()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([3.06134491e-04, 6.98709488e-03, 2.05003773e-04, 5.84280849e-01,\n",
       "       1.64828286e-01, 2.49455553e-02, 9.31214333e-01, 1.82469726e-01,\n",
       "       1.14941179e-06, 1.87841295e-06, 6.21517120e-06, 3.36545929e-02,\n",
       "       9.11137998e-01, 1.58931971e-01, 4.83539443e-05, 4.28070877e-07,\n",
       "       3.00293237e-07, 5.77240962e-06, 9.24921042e-05, 8.03018265e-05,\n",
       "       6.75234560e-06, 1.51504818e-02, 9.00563836e-01, 7.20365345e-01,\n",
       "       2.82101333e-03, 2.71354802e-04, 4.04667575e-04, 3.55769735e-05,\n",
       "       5.11673152e-01, 9.09208655e-01, 4.18880075e-01, 1.92401145e-04,\n",
       "       6.73088187e-04, 1.24309488e-04, 3.37590754e-01, 8.98752034e-01,\n",
       "       1.05692144e-03, 2.33176932e-01, 7.19539583e-01, 4.54376757e-01,\n",
       "       1.41463075e-02, 2.52352445e-04, 1.27090991e-03, 1.09135429e-03,\n",
       "       4.91683713e-05, 4.62389949e-07, 7.72917628e-01, 3.66144180e-01,\n",
       "       2.52536323e-04, 8.74981657e-02, 8.20681810e-01, 7.41231143e-01,\n",
       "       3.01656965e-02, 1.63271825e-03, 1.38260238e-03, 5.91513162e-05,\n",
       "       4.05433821e-06, 5.93505085e-01, 8.20710361e-01, 1.88775492e-04,\n",
       "       1.22357026e-01, 7.88423479e-01, 6.85073197e-01, 5.74627072e-02,\n",
       "       2.98262318e-03, 5.02652489e-03, 4.56930517e-04, 8.71223867e-01,\n",
       "       7.54958749e-01, 5.13532870e-02, 8.70302856e-01, 8.67553055e-01,\n",
       "       1.12581357e-01, 6.62332594e-01, 9.65456888e-02, 1.62648957e-03,\n",
       "       8.81560028e-01, 7.49024630e-01, 9.58382952e-05, 1.87657382e-02,\n",
       "       6.34016454e-01, 8.30893636e-01, 5.82906246e-01, 7.32933581e-01,\n",
       "       8.06817174e-01, 7.52889693e-01, 1.26662329e-01, 4.46536229e-04,\n",
       "       6.50741577e-01, 1.10612109e-01, 8.46632794e-02, 9.26767468e-01,\n",
       "       6.22468054e-01, 6.56636283e-02, 3.16250592e-01, 8.45368624e-01,\n",
       "       3.15682858e-01, 4.13068473e-01, 2.28649855e-01, 1.90662286e-05,\n",
       "       6.00638520e-03, 7.54870355e-01, 8.28795314e-01, 3.22087742e-02,\n",
       "       7.10824370e-01, 6.74151838e-01, 2.84326911e-01, 2.52669811e-01,\n",
       "       5.08263227e-08, 1.16334456e-06, 3.05625319e-04, 3.62610966e-01,\n",
       "       9.59836900e-01, 8.94935429e-02, 4.05862518e-02, 4.79843199e-01,\n",
       "       8.36108744e-01, 7.94858992e-01, 2.08037794e-01, 3.68214965e-01,\n",
       "       1.68030247e-01, 1.67653070e-07, 6.02369255e-06, 2.10248083e-02,\n",
       "       9.32036698e-01, 4.58786696e-01, 1.97643582e-02, 2.95089185e-01,\n",
       "       7.28614807e-01, 2.01153278e-01, 5.72622776e-01, 5.13686597e-01,\n",
       "       4.81894391e-09, 1.90886169e-08, 5.22123037e-06, 9.33106914e-02,\n",
       "       9.37285006e-01, 6.05429053e-01, 1.08566945e-02, 1.57708570e-01,\n",
       "       6.36566699e-01, 5.10453701e-01, 7.71752000e-01, 6.11257076e-01,\n",
       "       4.39473599e-01, 2.53301469e-09, 7.64204458e-07, 4.21468988e-02,\n",
       "       8.31775486e-01, 4.52521384e-01, 8.11535120e-01, 6.09454028e-02,\n",
       "       6.34820700e-01, 7.60428548e-01, 5.72470009e-01, 2.93340683e-01,\n",
       "       2.36140609e-01, 7.49058245e-07, 6.94175214e-02, 7.98171163e-01,\n",
       "       4.78865439e-03, 2.76032895e-01, 9.20222759e-01, 3.49713624e-01,\n",
       "       2.87000358e-01, 4.96368408e-01, 1.94790393e-01, 3.41325015e-01,\n",
       "       2.03221381e-01, 1.99373528e-01, 2.20203983e-05, 1.15604117e-03,\n",
       "       9.25937435e-04, 6.62886523e-05, 4.94562628e-05, 1.92293338e-02,\n",
       "       8.64507437e-01, 9.04606402e-01, 2.00409815e-01, 7.21703395e-02,\n",
       "       7.56389499e-02, 1.82438299e-01, 1.64479843e-05, 5.03279225e-05,\n",
       "       4.46407830e-05, 8.76090256e-04, 9.38206688e-02, 6.32483065e-01,\n",
       "       8.25538099e-01, 5.10869324e-01, 6.43070519e-01, 8.77537549e-01,\n",
       "       4.29188162e-01, 3.23036939e-01, 2.87316016e-05, 2.65343417e-03,\n",
       "       2.72683620e-01, 4.92789209e-01, 6.13632016e-02, 1.28742158e-01,\n",
       "       7.19319224e-01, 5.95492721e-01, 7.60326087e-01, 6.03691101e-01,\n",
       "       1.37033015e-01, 1.16292061e-02, 6.52934760e-02, 8.54719758e-01,\n",
       "       8.06752205e-01, 1.85490623e-01, 2.54828244e-01, 3.81787121e-01,\n",
       "       1.47706881e-01, 1.76302565e-03, 1.41513318e-01, 5.68908334e-01,\n",
       "       4.08244729e-01, 2.55342156e-01, 5.55442572e-01, 7.25027204e-01,\n",
       "       4.51895177e-01, 3.92854124e-01, 2.04012260e-01, 5.27803095e-05,\n",
       "       4.48234379e-03, 1.07214823e-01, 2.79343426e-01, 4.11355287e-01,\n",
       "       5.99163413e-01, 3.71335953e-01, 3.53959549e-05, 1.86701037e-03,\n",
       "       6.05851233e-01, 5.40688038e-01, 2.19378889e-01, 2.05526538e-02,\n",
       "       3.69939835e-06, 4.81210358e-04, 1.09552154e-02, 5.71832135e-02,\n",
       "       4.21030104e-01, 7.63890326e-01, 2.99015701e-01, 4.75632514e-05,\n",
       "       2.66216695e-03, 1.79687515e-02, 4.27606627e-02, 1.16607159e-01,\n",
       "       2.12978929e-01, 1.10006986e-04, 1.06508750e-03, 1.39309489e-03],\n",
       "      dtype=float32)"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.load(r'wifi_track_data\\dacang\\train_data\\rewards_deep_irl_fc_epoch10000_0122.npy')[-1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "p = np.array([[0,1],[0.5,0.5],[1,0]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[0. ],\n",
       "        [1. ]],\n",
       "\n",
       "       [[0.5],\n",
       "        [0.5]],\n",
       "\n",
       "       [[1. ],\n",
       "        [0. ]]])"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "new_p = p[:,:,np.newaxis]\n",
    "new_p"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[  1,   1,   0],\n",
       "        [  0,   0,   0]],\n",
       "\n",
       "       [[  0,   0,  10],\n",
       "        [100,   0,   3]],\n",
       "\n",
       "       [[  1,   2,   0],\n",
       "        [  0,   1,   1]]])"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "d = np.array([[[1,0,1],[0,100,0]],\n",
    "              [[1,0,2],[0, 0, 1]],\n",
    "              [[0,10,0],[0, 3, 1]]])\n",
    "d_swapped = np.transpose(d, (2, 1, 0))\n",
    "d_swapped"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[ 0. ,  0. ,  0. ],\n",
       "        [ 0. ,  0. ,  0. ]],\n",
       "\n",
       "       [[ 0. ,  0. ,  5. ],\n",
       "        [50. ,  0. ,  1.5]],\n",
       "\n",
       "       [[ 1. ,  2. ,  0. ],\n",
       "        [ 0. ,  0. ,  0. ]]])"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result = p[:,:,np.newaxis]*d_swapped\n",
    "result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([[ 0.0000,  0.0000,  0.0000],\n",
       "        [50.0000,  0.0000,  6.5000],\n",
       "        [ 1.0000,  2.0000,  0.0000]], dtype=torch.float64)"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x = result.sum(1)\n",
    "x = torch.from_numpy(x)\n",
    "x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [],
   "source": [
    "initial = np.array([0.3,0.7,0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [],
   "source": [
    "initial = torch.from_numpy(initial)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.Size([3])"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "initial.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [],
   "source": [
    "mu = initial.repeat(2,1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([[0.3000, 0.7000, 0.0000],\n",
       "        [0.3000, 0.7000, 0.0000]], dtype=torch.float64)"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mu_np "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([[ 0.3000,  0.7000,  0.0000],\n",
       "        [35.0000,  0.0000,  4.5500]], dtype=torch.float64)"
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "for i in range(1,2):\n",
    "    mu[i,:] = mu[i-1,:]@x\n",
    "mu"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([[ 0.3000,  0.7000,  0.0000],\n",
       "        [35.0000,  0.0000,  4.5500]], dtype=torch.float64)"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([1.3000, 2.7000, 1.0000], dtype=torch.float64)"
      ]
     },
     "execution_count": 65,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mu.sum(0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1"
      ]
     },
     "execution_count": 67,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "c = (1,2)\n",
    "c[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [],
   "source": [
    "reward_mat = np.array([[ 0, 0, 1, 1, 1, 0, 1, 0, 0, 2],\n",
    "                   [ 0, 0, 1, 2, 1, 0, 0, 0, 1, 1],\n",
    "                   [ 0,-1, 0, 1, 0, 0, 0, 0, 0, 1],\n",
    "                   [-1,-1,-1, 0, 0, 0, 0, 0, 0, 0],\n",
    "                   [ 0,-1, 0, 0, 0, 1, 0, 0, 0, 0],\n",
    "                   [ 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],\n",
    "                   [ 0, 0, 1, 1, 1, 2, 1, 1, 0, 0],\n",
    "                   [ 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],\n",
    "                   [ 0, 0, 0, 0, 0, 1, 0, 0, 0,-1],\n",
    "                   [ 0, 0, 0, 0, 0, 0, 0, 0,-1,-2],])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(100,)"
      ]
     },
     "execution_count": 71,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "reward_mat.reshape(100).shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.plotly.v1+json": {
       "config": {
        "plotlyServerURL": "https://plot.ly"
       },
       "data": [
        {
         "type": "scatter",
         "x": [
          0,
          1
         ],
         "y": [
          0,
          1
         ]
        }
       ],
       "frames": [
        {
         "data": [
          {
           "type": "scatter",
           "x": [
            1,
            2
           ],
           "y": [
            1,
            2
           ]
          }
         ]
        },
        {
         "data": [
          {
           "type": "scatter",
           "x": [
            1,
            4
           ],
           "y": [
            1,
            4
           ]
          }
         ]
        },
        {
         "data": [
          {
           "type": "scatter",
           "x": [
            3,
            4
           ],
           "y": [
            3,
            4
           ]
          }
         ],
         "layout": {
          "title": {
           "text": "End Title"
          }
         }
        }
       ],
       "layout": {
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "#2a3f5f"
            },
            "error_y": {
             "color": "#2a3f5f"
            },
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "bar"
           }
          ],
          "barpolar": [
           {
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "barpolar"
           }
          ],
          "carpet": [
           {
            "aaxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "baxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "heatmapgl": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmapgl"
           }
          ],
          "histogram": [
           {
            "marker": {
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "parcoords"
           }
          ],
          "pie": [
           {
            "automargin": true,
            "type": "pie"
           }
          ],
          "scatter": [
           {
            "fillpattern": {
             "fillmode": "overlay",
             "size": 10,
             "solidity": 0.2
            },
            "type": "scatter"
           }
          ],
          "scatter3d": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "#EBF0F8"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "#C8D4E3"
             },
             "line": {
              "color": "white"
             }
            },
            "type": "table"
           }
          ]
         },
         "layout": {
          "annotationdefaults": {
           "arrowcolor": "#2a3f5f",
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "autotypenumbers": "strict",
          "coloraxis": {
           "colorbar": {
            "outlinewidth": 0,
            "ticks": ""
           }
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "#8e0152"
            ],
            [
             0.1,
             "#c51b7d"
            ],
            [
             0.2,
             "#de77ae"
            ],
            [
             0.3,
             "#f1b6da"
            ],
            [
             0.4,
             "#fde0ef"
            ],
            [
             0.5,
             "#f7f7f7"
            ],
            [
             0.6,
             "#e6f5d0"
            ],
            [
             0.7,
             "#b8e186"
            ],
            [
             0.8,
             "#7fbc41"
            ],
            [
             0.9,
             "#4d9221"
            ],
            [
             1,
             "#276419"
            ]
           ],
           "sequential": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ]
          },
          "colorway": [
           "#636efa",
           "#EF553B",
           "#00cc96",
           "#ab63fa",
           "#FFA15A",
           "#19d3f3",
           "#FF6692",
           "#B6E880",
           "#FF97FF",
           "#FECB52"
          ],
          "font": {
           "color": "#2a3f5f"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "#E5ECF6",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "white"
          },
          "hoverlabel": {
           "align": "left"
          },
          "hovermode": "closest",
          "mapbox": {
           "style": "light"
          },
          "paper_bgcolor": "white",
          "plot_bgcolor": "#E5ECF6",
          "polar": {
           "angularaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "radialaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "yaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "zaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           }
          },
          "shapedefaults": {
           "line": {
            "color": "#2a3f5f"
           }
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "baxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "caxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          }
         }
        },
        "title": {
         "text": "Start Title"
        },
        "updatemenus": [
         {
          "buttons": [
           {
            "args": [
             null
            ],
            "label": "Play",
            "method": "animate"
           }
          ],
          "type": "buttons"
         }
        ],
        "xaxis": {
         "autorange": false,
         "range": [
          0,
          5
         ]
        },
        "yaxis": {
         "autorange": false,
         "range": [
          0,
          5
         ]
        }
       }
      }
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import plotly.graph_objects as go\n",
    "\n",
    "fig = go.Figure(\n",
    "    data=[go.Scatter(x=[0, 1], y=[0, 1])],\n",
    "    layout=go.Layout(\n",
    "        xaxis=dict(range=[0, 5], autorange=False),\n",
    "        yaxis=dict(range=[0, 5], autorange=False),\n",
    "        title=\"Start Title\",\n",
    "        updatemenus=[dict(\n",
    "            type=\"buttons\",\n",
    "            buttons=[dict(label=\"Play\",\n",
    "                          method=\"animate\",\n",
    "                          args=[None])])]\n",
    "    ),\n",
    "    frames=[go.Frame(data=[go.Scatter(x=[1, 2], y=[1, 2])]),\n",
    "            go.Frame(data=[go.Scatter(x=[1, 4], y=[1, 4])]),\n",
    "            go.Frame(data=[go.Scatter(x=[3, 4], y=[3, 4])],\n",
    "                     layout=go.Layout(title_text=\"End Title\"))]\n",
    ")\n",
    "\n",
    "fig.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "bigdata",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.16"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

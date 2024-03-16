import numpy as np
import cv2
import os
from PIL import Image

class Plotter():
    def __init__(self, sim):
        self.sim = sim
        self.img_size = 700
        self.scale = self.img_size / self.sim.map.size

        # open evtol icon
        self.evtol_icon_size = 24
        evtol_icon = Image.open('simulator/evtol_icon.png')
        evtol_icon = evtol_icon.resize((self.evtol_icon_size, self.evtol_icon_size))
        evtol_icon = np.array(evtol_icon)[:, :, 3]
        evtol_icon = np.stack([evtol_icon, evtol_icon, evtol_icon], axis=-1)
        self.evtol_icon = 255 - evtol_icon

        os.makedirs('./simulator/frames', exist_ok=True)


    def plot(self):
        # create an image of size contained in map
        img = 255 * np.ones((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        if type(self.sim.map).__name__ == 'SatMap':
            # plot satellite image
            sat = cv2.resize(self.sim.map.sat, (self.img_size, self.img_size))
            img = sat.astype(np.uint8)

        # plot vertiports
        for vertiport in self.sim.map.vertiports:
            cv2.circle(img, (int(self.scale * vertiport.x), int(self.scale * vertiport.y)), radius=5, color=(0, 0, 255), thickness=-1)  # red circle
            cv2.putText(img, f'{len(vertiport.cur_passengers)}', (int(self.scale * vertiport.x), int(self.scale * vertiport.y)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            # plot vp id
            # cv2.putText(img, f'{vertiport.id}', (int(self.scale * vertiport.x), int(self.scale * vertiport.y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # plot agents
        for agent in self.sim.agents:
            color = (255, 0, 0)
            if agent.grounded:
                color = (0, 255, 0)
            cv2.circle(img, (int(self.scale * agent.x), int(self.scale * agent.y)), radius=5, color=color, thickness=-1)  # blue circle
            # put ring of size 5 around agent
            # cv2.circle(img, (int(self.scale * agent.x), int(self.scale * agent.y)), radius=int(self.scale*5), color=(0, 0, 0), thickness=1)
            # cv2.putText(img, f'{agent.id}', (int(self.scale * agent.x), int(self.scale * agent.y)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            # M = cv2.getRotationMatrix2D((self.evtol_icon_size//2, self.evtol_icon_size//2), np.degrees(agent.theta), 1)
            # evtol_icon = cv2.warpAffine(self.evtol_icon, M, (self.evtol_icon_size, self.evtol_icon_size))
            # img[int(self.scale * agent.y - self.evtol_icon_size//2):int(self.scale * agent.y + self.evtol_icon_size//2), int(self.scale * agent.x - self.evtol_icon_size//2):int(self.scale * agent.x + self.evtol_icon_size//2)] = evtol_icon

        # add time to lower right
        cv2.putText(img, f'Time: {self.sim.time}', (int(self.img_size * 0.02), int(self.img_size * 0.07)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # cv2.imwrite(f'./simulator/frames/frame_{self.sim.time}.png', img)
        cv2.imshow('Simulation', img)
        cv2.waitKey(1)
        # pause utnil key is pressed
        # cv2.waitKey(0)

        cv2.destroyAllWindows()
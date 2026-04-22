# E-Waste Dataset > Fix annotations of some bar-phones incorrectly labelled as smartphones
https://universe.roboflow.com/electronic-waste-detection/e-waste-dataset-r0ojc

Provided by a Roboflow user
License: CC BY 4.0

# Overview
The goal of this project was to create a structured dataset which can be used to train computer vision models to detect electronic waste devices, i.e., e-waste or Waste Electrical and Electronic Equipment (WEEE). Due to the often-subjective differences between e-waste and functioning electronic devices, a model trained on this dataset could also be used to detect electronic devices in general. However, it must be noted that for the purposes of e-waste recognition, this dataset does not differentiate between different brands or models of the same type of electronic devices, e.g. smartphones, and it also includes images of damaged equipment.

The structure of this dataset is based on the UNU-KEYS classification [Wang et al., 2012](https://ieeexplore.ieee.org/document/6360480), [Forti et al., 2018](https://collections.unu.edu/eserv/UNU:6477/RZ_EWaste_Guidelines_LoRes.pdf). Each class in this dataset has a tag containing its corresponding UNU-KEY. This dataset structure has the following benefits:
1. It allows the user to easily classify e-waste devices regardless of which e-waste definition their country or organization uses, thanks to the correlation between the UNU-KEYS and other classifications such as the HS-codes or the EU-6 categories, defined in the WEEE directive;
2. It helps dataset contributors focus on adding e-waste devices with higher priority compared to arbitrarily chosen devices. This is because electronic devices in the same UNU-KEY category have similar function, average weight and life-time distribution as well as comparable material composition, both in terms of hazardous substances and valuable materials, and related end-of-life attributes [Forti et al., 2018](https://collections.unu.edu/eserv/UNU:6477/RZ_EWaste_Guidelines_LoRes.pdf).
3. It gives dataset contributors a clear goal of which electronic devices still need to be added and a clear understanding of their progress in the seemingly endless task of creating an e-waste dataset.

This dataset contains annotated images of e-waste from every UNU-KEY category. According to [Forti et al., 2018](https://collections.unu.edu/eserv/UNU:6477/RZ_EWaste_Guidelines_LoRes.pdf), there are a total of 54 UNU-KEY e-waste categories.

# Description of Classes
At the time of writing, 22. Apr. 2024, the dataset has 19613 annotated images and 77 classes. The dataset has mixed bounding-box and polygon annotations. Each class of the dataset represents one type of electronic device. Different models of the same type of device belong to the same class. For example, different brands of smartphones are labelled as "Smartphone", regardless of their make or model. Many classes can belong to the same UNU-KEY category and therefore have the same tag. For example, the classes "Smartphone" and "Bar-Phone" both belong to the UNU-KEY category "0306 - Mobile Phones". The images in the dataset are anonymized, meaning that no people were annotated and images containing visible faces were removed.

The dataset was almost entirely built by cloning annotated images from the following open-source Roboflow datasets: [1]-[91]. Some of the images in the dataset were acquired from the Wikimedia Commons website. Those images were chosen to have an unrestrictive license, i.e., they belong to the public domain. They were manually annotated and added to the dataset.

# Cite This Project
This work was done as part of the PhD of Dimitar Iliev, student at the Faculty of German Engineering and Industrial Management at the Technical University of Sofia, Bulgaria and in collaboration with the Faculty of Computer Science at Otto-von-Guericke-University Magdeburg, Germany.

If you use this dataset in a research paper, please cite it using the following BibTeX:
```
@article{iliev2024EwasteDataset,
  author  = "Iliev, Dimitar and Marinov, Marin and Ortmeier, Frank",
  title   = "A proposal for a new e-waste image dataset based on the unu-keys classification",
  journal = "XXIII-rd International Symposium on Electrical Apparatus and Technologies SIELA 2024",
  year    = 2024,
  volume  = "23",
  number  = "to appear",
  pages    = {to appear}
  note    = {under submission}
}
```

# Contribution Guidelines
## Image Collection
1. Choose a specific electronic device type to add to the dataset and find its corresponding UNU-KEY.
		*  The chosen type of device should have a characteristic design which an object detection model can learn. For example, CRT monitors look distinctly different than flat panel monitors and should therefore belong to a different class, regardless that they are both monitors. In contrast, LED monitors and LCD monitors look very similar and are therefore both labelled as Flat-Panel-Monitor in this dataset.
3. Collect images of this type of device.
		*  Take note of the license of those images and their author/s to avoid copyright infringement. 
		*  Do not collect images with visible faces to protect personal data and comply with GDPR regulations.
		*  Images can be collected by cloning other open source Roboflow datasets, or by downloading them from other websites.
		*  It is recommended to collect a minimum of 100 images and a maximum of 1000 images per class.
4. Anonymize the data.

## Labelling
1. The name of the labelling class must describe the device type and be written with a capital first letter.
		*  If the device type is described with more than one word, the first letter of each word must be capitalized and the words must be separated with a dash, for example: Flat-Panel-Monitor.
2. A tag must be applied to all images of a given class. The tag must contain the UNU-KEY category corresponding to that type of device. A tag for each of the 54 UNU-KEY categories already exists in the dataset and can easily be applied by selecting all of the images of a given class and clicking on the "Apply tags" button.
## Annotation
1. Objects can be annotated either with a bounding-box or polygon annotation. 
2. Annotations of irrelevant objects must be deleted.

## Citation
If the licence of the collected images require that the author is cited, contributors can do this by adding their sources to the "List of Image Sources" at the end of this document.


# References
Wang, F., Huisman, J., Baldé, K., Stevels, A. (2012). A systematic and compatible classification of WEEE. In Electronics Goes Green, Berlin, Germany. Available: https://ieeexplore.ieee.org/document/6360480

Vanessa Forti, Kees Baldé, and Ruediger Kuehr, (2018). ‘E-waste Statistics: Guidelines on Classifications, Reporting and Indicators, second edition.’, United Nations University, ViE – SCYCLE, Bonn, Germany. Available: https://collections.unu.edu/eserv/UNU:6477/RZ_EWaste_Guidelines_LoRes.pdf


# List of Image Sources
[1] capstone-wyicl, ‘20_PMD Dataset’, Roboflow. Accessed: Apr. 12, 2024. [Online]. Available: https://universe.roboflow.com/capstone-wyicl/20_pmd
[2] SO, ‘all_finalize Dataset’, Roboflow. Accessed: Apr. 08, 2024. [Online]. Available: https://universe.roboflow.com/so-d4hcz/all_finalize
[3] razikous, ‘ARapp02 Dataset’, Roboflow. Accessed: Apr. 10, 2024. [Online]. Available: https://universe.roboflow.com/razikous/arapp02
[4] blank, ‘bike Dataset’, Roboflow. Accessed: Apr. 12, 2024. [Online]. Available: https://universe.roboflow.com/blank/bike-coppr
[5] xx2001xx, ‘BlindSight-Dataset Dataset’, Roboflow. Accessed: Apr. 10, 2024. [Online]. Available: https://universe.roboflow.com/xx2001xx/blindsight-dataset
[6] project-1iwh1, ‘boiler Dataset’, Roboflow. Accessed: Apr. 02, 2024. [Online]. Available: https://universe.roboflow.com/project-1iwh1/boiler
[7] Boiler modeling, ‘Boilers Dataset’, Roboflow. Accessed: Apr. 02, 2024. [Online]. Available: https://universe.roboflow.com/boiler-modeling/boilers
[8] K. Teo, ‘BP Monitor Model Dataset’, Roboflow. Accessed: Apr. 12, 2024. [Online]. Available: https://universe.roboflow.com/keane-teo/bp-monitor-model
[9] project-1-ldanh, ‘Bulb Detection Dataset’, Roboflow. Accessed: Apr. 11, 2024. [Online]. Available: https://universe.roboflow.com/project-1-ldanh/bulb-detection
[10] Microsoft, ‘COCO Dataset’, Roboflow. Accessed: Mar. 26, 2024. [Online]. Available: https://universe.roboflow.com/microsoft/coco
[11] cmpu-kerempu-pxipo, ‘data Trainig Dataset’, Roboflow. Accessed: Apr. 12, 2024. [Online]. Available: https://universe.roboflow.com/cmpu-kerempu-pxipo/data-trainig
[12] vidya-f56fa, ‘dataset Dataset’, Roboflow. Accessed: Apr. 10, 2024. [Online]. Available: https://universe.roboflow.com/vidya-f56fa/dataset-7q5yp
[13] aimslab, ‘DeepLearning_20230430 Dataset’, Roboflow. Accessed: Apr. 11, 2024. [Online]. Available: https://universe.roboflow.com/aimslab/deeplearning_20230430
[14] aero-ml, ‘Detection Dataset’, Roboflow. Accessed: Apr. 10, 2024. [Online]. Available: https://universe.roboflow.com/aero-ml/detection-etcb3
[15] Emmanuel, ‘Detection Dataset’, Roboflow. Accessed: Apr. 08, 2024. [Online]. Available: https://universe.roboflow.com/emmanuel-yvctn/detection-a2gcb
[16] project-xrojb, ‘device_smoke_detector_1 Dataset’, Roboflow. Accessed: Apr. 12, 2024. [Online]. Available: https://universe.roboflow.com/project-xrojb/device_smoke_detector_1
[17] computervisionprojects-dfryr, ‘Digital-BP-kit Dataset > Overview’, Roboflow. Accessed: Apr. 12, 2024. [Online]. Available: https://universe.roboflow.com/computervisionprojects-dfryr/digital-bp-kit-exbid
[18] drone-rwsrk, ‘Drone Dataset’, Roboflow. Accessed: Apr. 12, 2024. [Online]. Available: https://universe.roboflow.com/drone-rwsrk/drone-cmxwz
[19] Circularity Space, ‘E-waste Dataset’, Roboflow. Accessed: Mar. 24, 2024. [Online]. Available: https://universe.roboflow.com/circularity-space-nbzxv/e-waste-alwua
[20] new-workspace-f7og7, ‘E-waste Dataset’, Roboflow. Accessed: Mar. 25, 2024. [Online]. Available: https://universe.roboflow.com/new-workspace-f7og7/e-waste-mx8fq
[21] ewaste-fkqc4, ‘E-Waste Detection Dataset’, Roboflow. Accessed: Mar. 25, 2024. [Online]. Available: https://universe.roboflow.com/ewaste-fkqc4/e-waste-detection-nqnkq/dataset/1
[22] P. Kumar, ‘E-waste detection Dataset’, Roboflow. Accessed: Mar. 25, 2024. [Online]. Available: https://universe.roboflow.com/prem-kumar-fhkau/e-waste-detection-5om83/dataset/1
[23] TRCProject, ‘E-waste detection model Dataset’, Roboflow. Accessed: Mar. 21, 2024. [Online]. Available: https://universe.roboflow.com/trcproject/e-waste-detection-model
[24] divelement-web-services, ‘Eddie’s Office POC Dataset’, Roboflow. Accessed: Apr. 12, 2024. [Online]. Available: https://universe.roboflow.com/divelement-web-services/eddie-s-office-poc
[25] V. Chebet, ‘Electroni Waste Dataset’, Roboflow. Accessed: Mar. 25, 2024. [Online]. Available: https://universe.roboflow.com/valentine-chebet-bkd8i/electroni-waste
[26] senior-design-yolov7, ‘Equipment Data Dataset’, Roboflow. Accessed: Apr. 13, 2024. [Online]. Available: https://universe.roboflow.com/senior-design-yolov7/equipment-data
[27] fall-risk-detection, ‘Fall Risk Object Detection Dataset’, Roboflow. Accessed: Apr. 11, 2024. [Online]. Available: https://universe.roboflow.com/fall-risk-detection/fall-risk-object-detection
[28] fall-risk-detection, ‘Fallrisk Version 2 Dataset’, Roboflow. Accessed: Apr. 11, 2024. [Online]. Available: https://universe.roboflow.com/fall-risk-detection/fallrisk-version-2
[29] interior-classification, ‘Faysal 2+3 Dataset’, Roboflow. Accessed: Apr. 10, 2024. [Online]. Available: https://universe.roboflow.com/interior-classification/faysal-2-3
[30] sdg-zpesj, ‘Fluorescent light detection Dataset’, Roboflow. Accessed: Apr. 03, 2024. [Online]. Available: https://universe.roboflow.com/sdg-zpesj/fluorescent-light-detection
[31] A. Kutit, ‘Fridges Dataset’, Roboflow. Accessed: Apr. 09, 2024. [Online]. Available: https://universe.roboflow.com/ayham-kutit/fridges
[32] furncomp, ‘Furniture Focused Indoor Dataset Dataset’, Roboflow. Accessed: Apr. 08, 2024. [Online]. Available: https://universe.roboflow.com/furncomp/furniture-focused-indoor-dataset
[33] glucometerocr, ‘Glucometer Dataset’, Roboflow. Accessed: Apr. 12, 2024. [Online]. Available: https://universe.roboflow.com/glucometerocr/glucometer
[34] swleeyg-gmail-com, ‘Guitar Boogie Dataset’, Roboflow. Accessed: Apr. 10, 2024. [Online]. Available: https://universe.roboflow.com/swleeyg-gmail-com/guitar-boogie
[35] guitars, ‘Guitar Detection 2.0 Dataset’, Roboflow. Accessed: Apr. 10, 2024. [Online]. Available: https://universe.roboflow.com/guitars/guitar-detection-2.0
[36] RAIN, ‘house hold objects_yolov5 Dataset’, Roboflow. Accessed: Apr. 08, 2024. [Online]. Available: https://universe.roboflow.com/rain-tbklt/house-hold-objects_yolov5
[37] sut-0zrfh, ‘household items detection 2 Dataset’, Roboflow. Accessed: Apr. 03, 2024. [Online]. Available: https://universe.roboflow.com/sut-0zrfh/household-items-detection-2
[38] pembelajaranmesin-38zuw, ‘household tools Dataset’, Roboflow. Accessed: Apr. 03, 2024. [Online]. Available: https://universe.roboflow.com/pembelajaranmesin-38zuw/household-tools
[39] energy-chaser, ‘illumination Dataset’, Roboflow. Accessed: Apr. 11, 2024. [Online]. Available: https://universe.roboflow.com/energy-chaser/illumination
[40] daffodil, ‘ilviimage Dataset’, Roboflow. Accessed: Apr. 12, 2024. [Online]. Available: https://universe.roboflow.com/daffodil/ilviimage
[41] Universidad de los Andes, ‘Instrument-Lab Detection Dataset’, Roboflow. Accessed: Apr. 13, 2024. [Online]. Available: https://universe.roboflow.com/universidad-de-los-andes-lzihh/instrument-lab-detection
[42] first-cuskb, ‘kona_2.11 Dataset’, Roboflow. Accessed: Apr. 08, 2024. [Online]. Available: https://universe.roboflow.com/first-cuskb/kona_2.11
[43] lamp-axx7h, ‘lamp Dataset’, Roboflow. Accessed: Apr. 11, 2024. [Online]. Available: https://universe.roboflow.com/lamp-axx7h/lamp-akkc0
[44] M. Ojeda, ‘Light bulb Dataset’, Roboflow. Accessed: Apr. 03, 2024. [Online]. Available: https://universe.roboflow.com/mateo-ojeda/light-bulb-hqx1h
[45] sdg-zpesj, ‘lightbulb detection Dataset’, Roboflow. Accessed: Apr. 03, 2024. [Online]. Available: https://universe.roboflow.com/sdg-zpesj/lightbulb-detection
[46] mirza-w9dcj, ‘lightpost Dataset’, Roboflow. Accessed: Apr. 11, 2024. [Online]. Available: https://universe.roboflow.com/mirza-w9dcj/lightpost-auyso
[47] bsbi, ‘medical equipment detection Dataset’, Roboflow. Accessed: Apr. 12, 2024. [Online]. Available: https://universe.roboflow.com/bsbi/medical-equipment-detection
[48] visi, ‘Memory Dataset’, Roboflow. Accessed: Oct. 10, 2023. [Online]. Available: https://universe.roboflow.com/visi-bdzkk/memory
[49] test-ip1lx, ‘meter Dataset’, Roboflow. Accessed: Apr. 12, 2024. [Online]. Available: https://universe.roboflow.com/test-ip1lx/meter-x51wi
[50] my-workspace-rve0x, ‘moveable objects Dataset’, Roboflow. Accessed: Apr. 08, 2024. [Online]. Available: https://universe.roboflow.com/my-workspace-rve0x/moveable-objects
[51] mower, ‘Mower Dataset’, Roboflow. Accessed: Apr. 12, 2024. [Online]. Available: https://universe.roboflow.com/mower/mower-u0vh2
[52] mower-gyzd2, ‘mower Dataset’, Roboflow. Accessed: Apr. 11, 2024. [Online]. Available: https://universe.roboflow.com/mower-gyzd2/mower-2r1ou
[53] A. Altymyshev, ‘napitki_refs Dataset’, Roboflow. Accessed: Apr. 09, 2024. [Online]. Available: https://universe.roboflow.com/aliym-altymyshev-hshww/napitki_refs
[54] object-detection-cjnfr, ‘Object Detection Challenge Dataset’, Roboflow. Accessed: Apr. 10, 2024. [Online]. Available: https://universe.roboflow.com/object-detection-cjnfr/object-detection-challenge
[55] object-detection-0xgp1, ‘object detection Dataset’, Roboflow. Accessed: Apr. 10, 2024. [Online]. Available: https://universe.roboflow.com/object-detection-0xgp1/object-detection-rdhtd
[56] objectdetection-bekit, ‘object detection project Dataset’, Roboflow. Accessed: Apr. 08, 2024. [Online]. Available: https://universe.roboflow.com/objectdetection-bekit/object-detection-project-70quh
[57] Universiti Malaysia Pahang, ‘objectdetection Dataset’, Roboflow. Accessed: Apr. 13, 2024. [Online]. Available: https://universe.roboflow.com/universiti-malaysia-pahang-qcvas/objectdetection-ngxjp
[58] ajmodel, ‘office dataset internal v2 Dataset’, Roboflow. Accessed: Apr. 10, 2024. [Online]. Available: https://universe.roboflow.com/ajmodel/office-dataset-internal-v2
[59] OMO, ‘omo52 Dataset’, Roboflow. Accessed: Apr. 03, 2024. [Online]. Available: https://universe.roboflow.com/omo/omo52
[60] optimizedai-hqilj, ‘OptimizedOfficeAI_add Dataset’, Roboflow. Accessed: Apr. 10, 2024. [Online]. Available: https://universe.roboflow.com/optimizedai-hqilj/optimizedofficeai_add
[61] roboticslearning, ‘Oscilloscope Detection3 Dataset’, Roboflow. Accessed: Apr. 13, 2024. [Online]. Available: https://universe.roboflow.com/roboticslearning/oscilloscope-detection3
[62] energy-chaser, ‘Ovens Dataset’, Roboflow. Accessed: Apr. 04, 2024. [Online]. Available: https://universe.roboflow.com/energy-chaser/ovens
[63] nikki-foysal, ‘Phase1+2+3_v3 Dataset’, Roboflow. Accessed: Apr. 10, 2024. [Online]. Available: https://universe.roboflow.com/nikki-foysal/phase1-2-3_v3
[64] school-59kbs, ‘project Dataset’, Roboflow. Accessed: Apr. 08, 2024. [Online]. Available: https://universe.roboflow.com/school-59kbs/project-djju2
[65] research-9govy, ‘Reface and Sliders 3 Dataset’, Roboflow. Accessed: Apr. 10, 2024. [Online]. Available: https://universe.roboflow.com/research-9govy/reface-and-sliders-3
[66] universitas-krisnadwipayana, ‘Smart Home Dataset’, Roboflow. Accessed: Apr. 11, 2024. [Online]. Available: https://universe.roboflow.com/universitas-krisnadwipayana/smart-home
[67] tud-mq613, ‘smoke detectors Dataset’, Roboflow. Accessed: Apr. 12, 2024. [Online]. Available: https://universe.roboflow.com/tud-mq613/smoke-detectors
[68] polytechnic-university-ubjwp, ‘solar panal Dataset’, Roboflow. Accessed: Apr. 02, 2024. [Online]. Available: https://universe.roboflow.com/polytechnic-university-ubjwp/solar-panal
[69] I. Bunescu, ‘Specific Electronics Challenge v2 Dataset’, Roboflow. Accessed: Apr. 10, 2024. [Online]. Available: https://universe.roboflow.com/iulia-bunescu-vldcs/specific-electronics-challenge-v2
[70] step5-v3, ‘step5v4 Dataset’, Roboflow. Accessed: Apr. 12, 2024. [Online]. Available: https://universe.roboflow.com/step5-v3/step5v4
[71] street-lamps, ‘street lamps Dataset’, Roboflow. Accessed: Apr. 11, 2024. [Online]. Available: https://universe.roboflow.com/street-lamps/street-lamps-dpeqc
[72] siho, ‘street-lamp(1) Dataset’, Roboflow. Accessed: Apr. 11, 2024. [Online]. Available: https://universe.roboflow.com/siho/street-lamp-1
[73] synth204, ‘synth199 Dataset’, Roboflow. Accessed: Apr. 10, 2024. [Online]. Available: https://universe.roboflow.com/synth204/synth199
[74] bhagath-nairb, ‘Table lamp Dataset’, Roboflow. Accessed: Apr. 12, 2024. [Online]. Available: https://universe.roboflow.com/bhagath-nairb/table-lamp/dataset/1
[75] final-projectboostcamp-ai-tech, ‘tablelamp dataset Dataset’, Roboflow. Accessed: Apr. 11, 2024. [Online]. Available: https://universe.roboflow.com/final-projectboostcamp-ai-tech/tablelamp-dataset
[76] object-detection-l7hk4, ‘talov-flashlight Dataset’, Roboflow. Accessed: Apr. 11, 2024. [Online]. Available: https://universe.roboflow.com/object-detection-l7hk4/talov-flashlight
[77] new-workspace-ej5xn, ‘targetv3 Dataset’, Roboflow. Accessed: Apr. 02, 2024. [Online]. Available: https://universe.roboflow.com/new-workspace-ej5xn/targetv3
[78] new-workspace-ej5xn, ‘targetv4 Dataset’, Roboflow. Accessed: Apr. 02, 2024. [Online]. Available: https://universe.roboflow.com/new-workspace-ej5xn/targetv4
[79] pruebas-de-200, ‘Telephone Dataset’, Roboflow. Accessed: Apr. 10, 2024. [Online]. Available: https://universe.roboflow.com/pruebas-de-200/telephone-5ozo2
[80] alexander437-gzzhf, ‘Tello_detect Dataset’, Roboflow. Accessed: Apr. 12, 2024. [Online]. Available: https://universe.roboflow.com/alexander437-gzzhf/tello_detect
[81] circularity-7fb36, ‘Tetrapak Dataset’, Roboflow. Accessed: Apr. 10, 2024. [Online]. Available: https://universe.roboflow.com/circularity-7fb36/tetrapak-quant
[82] tmtestes, ‘TM_MedicalEq Dataset’, Roboflow. Accessed: Apr. 12, 2024. [Online]. Available: https://universe.roboflow.com/tmtestes/tm_medicaleq
[83] rgbdmaterial, ‘TUM Yolov5 Dataset’, Roboflow. Accessed: Apr. 10, 2024. [Online]. Available: https://universe.roboflow.com/rgbdmaterial/tum-yolov5
[84] uas-maura-putri-camyla, ‘UAS CLOUD COMPUTING Dataset’, Roboflow. Accessed: Apr. 03, 2024. [Online]. Available: https://universe.roboflow.com/uas-maura-putri-camyla/uas-cloud-computing-g4zbt
[85] S. Poh, ‘updated Dataset’, Roboflow. Accessed: Apr. 03, 2024. [Online]. Available: https://universe.roboflow.com/shilin-poh/updated-tbwff
[86] workspace-hrqhs, ‘val_gehc_8 Dataset’, Roboflow. Accessed: Apr. 12, 2024. [Online]. Available: https://universe.roboflow.com/workspace-hrqhs/val_gehc_8
[87] project-yoa2s, ‘vvv1 Dataset’, Roboflow. Accessed: Apr. 09, 2024. [Online]. Available: https://universe.roboflow.com/project-yoa2s/vvv1
[88] energy-chaser, ‘Washing_Machines Dataset’, Roboflow. Accessed: Apr. 04, 2024. [Online]. Available: https://universe.roboflow.com/energy-chaser/washing_machines
[89] xmas-lights, ‘Xmas Lights Dataset’, Roboflow. Accessed: Apr. 11, 2024. [Online]. Available: https://universe.roboflow.com/xmas-lights/xmas-lights
[90] workspace-hrqhs, ‘xray, pms, defibrillator Dataset’, Roboflow. Accessed: Apr. 12, 2024. [Online]. Available: https://universe.roboflow.com/workspace-hrqhs/xray--pms--defibrillator
[91] ‘yolo Dataset’, Roboflow. Accessed: Apr. 11, 2024. [Online]. Available: https://universe.roboflow.com/screwdriver-3wsuv/yolo-fpnyc
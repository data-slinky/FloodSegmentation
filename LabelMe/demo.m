clear all

addpath(genpath('/Users/johnkimnguyen/Box Sync/FloodSegmentation/LabelMe'));

% Define the root folder for the images
HOMEIMAGES = '/Users/johnkimnguyen/Box Sync/FloodSegmentation/Flood_Labels2/Images/'; % you can set here your default folder
HOMEANNOTATIONS = '/Users/johnkimnguyen/Box Sync/FloodSegmentation/Flood_Labels2/Annotations/'; % you can set here your default folder

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OVERLAY THE SEGMENTS ON THEN IMAGE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% select one annotation file from one of the folders:
filename = fullfile(HOMEANNOTATIONS, 'test_image69.xml');
% read the image and annotation struct:
[annotation, img] = LMread(filename, HOMEIMAGES);

% plot the annotations
LMplot(annotation, img)

% example manipulations
% functions: LMimscale, LMimcrop, LMimpad and LMcookimage. LMcookimage is a
% [newannotation, newimg, crop, scaling, err, msg] = LMcookimage(annotation, img, 'maximagesize', [256 256], 'impad', 255);
% LMplot(newannotation, newimg)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OUTPUT THE SEGMENTS ONLY
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Create database from folder
database = LMdatabase(HOMEANNOTATIONS);

% Locate office scenes with only one screen. 
[D,j]  = LMquery(database, 'object.name', 'flood');length(j)

for n = 43:44
    w = D(n).annotation.imagesize;
    width = str2num(w.nrows);
    height = str2num(w.ncols);
    [pathstr, name, ext] = fileparts(D(n).annotation.filename);

    blankimage = zeros(width, height);
    [x,y] = LMobjectpolygon(D(n).annotation, 1);
    hFig = figure(1);
    imshow(blankimage)
    hold on
    fill(x{1}, y{1}, [0.98836199999999996 0.99836400000000003 0.64492400000000005])
    set(gca,'position',[0 0 1 1],'units','normalized')
    whitebg([0 0 0])
    axis off
    r = 150; % pixels per inch
    set(gcf, 'PaperUnits', 'inches', 'PaperPosition', [0 0 height width]/r);
    print(gcf,'-dpng',sprintf('-r%d',r), strcat('/Users/johnkimnguyen/Box Sync/FloodSegmentation/', name, '.png'));
    close
end

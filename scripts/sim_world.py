#!/usr/bin/env python2

import Box2D
from Box2D import (b2PolygonShape, b2World, b2Distance)
import cv2
import numpy as np
from matplotlib import cm
import random

import h5py

import skvideo.io

# import model_creation as mc

# MPL no display backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


cmap = cm.get_cmap('Spectral')

PPM = 10.0  # pixels per meter
SCREEN_WIDTH, SCREEN_HEIGHT = 256, 256
SCREEN_OFFSETX, SCREEN_OFFSETY = SCREEN_WIDTH/2, SCREEN_HEIGHT
BACKGROUND_COLOUR = 255
FOREGROUND_COLOUR = 1

CROP_H_SIZE = 16
CROP_SIZE = CROP_H_SIZE * 2

def engine2canvas_vertices(vertices):
    return [(int(SCREEN_OFFSETX + v[0]), int(SCREEN_OFFSETY - v[1]))
            for v in vertices]

def approxEqual(v1, v2, d = .0):
    return (abs(v1[0] - v2[0]) < d) and (abs(v1[1] - v2[1]) < d)

def _draw_contact(ce, screen, body):
    # print ce.worldManifold.normal gives a wrong different value from the code
    # below rather than accessing print ce.worldManifold an looking up the value
    a = ce.worldManifold
    normal_M = a.normal # default normal vector (God object vector to closest outer point)

    # Update normal to
    # To get actual vector of collision
    # see http://www.iforce2d.net/b2dtut/collision-anatomy

    for p in ce.worldManifold.points:
        if (p == Box2D.b2Vec2(0, 0)): # Skip when only one edge is hit
            continue
        vel1 = ce.fixtureA.body.GetLinearVelocityFromWorldPoint(p)
        vel2 = ce.fixtureB.body.GetLinearVelocityFromWorldPoint(p)

        impact_v = vel2 - vel1
        # if (not approxEqual(impact_v, Box2D.b2Vec2(0, 0), 0.1)): # Update normal if in collision
        normal_M = impact_v

        position = engine2canvas_vertices([(p[0] * PPM, p[1] * PPM)])[0]
        colour = (250, 15, 15)
        rad = 0.3
        cv2.circle(screen, position, int(rad * PPM), colour)
        # draw line across the normal vec
        len = 1.0
        endpoint = (int(position[0] + PPM * len * normal_M[0]),
                    int(position[1] + PPM * len * normal_M[1]))
        # print endpoint, position, normal_M
        cv2.line(screen, position, endpoint, colour)

def _draw_circle(circle, screen, body, fixture):
    position = engine2canvas_vertices([body.transform * circle.pos * PPM])[0]
    colour = body.userData['colour']
    cv2.circle(screen, position, int(circle.radius * PPM), colour)

def _draw_polygon(polygon, screen, body=None, fixture=None):
    transform = body.transform
    vertices = np.asarray(engine2canvas_vertices(
        [transform * v * PPM for v in polygon.vertices]))

    # Draw the contour and fill
    fill_b, fill_g, fill_r = body.userData["colour"]#128, 128, 256
    line_b, line_g, line_r = body.userData["colour_border"]
    thickness = body.userData["border_thickness"]

    # print vertices
    cv2.fillPoly(screen, [vertices], (fill_b, fill_g, fill_r))
    cv2.polylines(screen, [vertices], True, (line_b, line_g, line_r), thickness)

def _draw_edge(edge, screen, body, fixture):
        vertices = engine2canvas_vertices([body.transform * edge.vertex1 * PPM,
                                           body.transform * edge.vertex2 * PPM])
        cv2.line(screen, vertices[0], vertices[1], body.userData["colour"])

# Add extra functions
Box2D.b2CircleShape.draw = _draw_circle
Box2D.b2PolygonShape.draw = _draw_polygon
Box2D.b2EdgeShape.draw = _draw_edge
Box2D.b2Contact.draw = _draw_contact

def inBodyCollision(body):
    if (body.mass > .0):
        for contact_edge in body.contacts:
            if contact_edge.contact.touching and contact_edge.other.mass > .0:
                return True
    return False

def createBaseWorld():
    world = b2World(gravity=(0, -9.8), doSleep=True)
    data = {"colour": [169, 169, 169]}
    data.update({"colour_border": [128, 128, 128]})
    data.update({"border_thickness": 1})
    groundBody = world.CreateStaticBody(position=(0, -1),
                                        shapes=b2PolygonShape(box=(13, 2)), # half sizes
                                        angle=0.0,
                                        userData=data)
    groundBody2 = world.CreateStaticBody(position=(SCREEN_OFFSETX/PPM, SCREEN_OFFSETY/PPM/2.0),
                                        shapes=b2PolygonShape(box=(1, 13)), # half sizes
                                        angle=0,
                                        userData=data)
    groundBody3 = world.CreateStaticBody(position=(-SCREEN_OFFSETX/PPM, SCREEN_OFFSETY/PPM/2.0),
                                        shapes=b2PolygonShape(box=(1, 13)), # half sizes
                                        angle=0,
                                        userData=data)
    groundBody4 = world.CreateStaticBody(position=(0, SCREEN_OFFSETY/PPM),
                                        shapes=b2PolygonShape(box=(13, 1)), # half sizes
                                        angle=0.0,
                                        userData=data)
    return world

def createStaticPaddle(world, pos, angle):
    data = {"colour": [169, 169, 169]}
    data.update({"colour_border": [128, 128, 128]})
    data.update({"border_thickness": 1})
    paddleBody = world.CreateStaticBody(position=pos,
                                        angle=angle,
                                        shapes=b2PolygonShape(box=(3, 0.2)), # half sizes
                                        userData=data)
    return paddleBody

def createConvexBody(world, pos, angle=0,
                     density=1,
                     friction=0.5,
                     restitution=0.5,
                     vertices=[],
                     border_thickness=1):
    if not vertices:
        vertices = [(0,0), (0, 2), (2,2), (2, 0)] # (3,1),
    shape = b2PolygonShape(vertices=vertices)
    data = {"colour": np.asarray(cmap(random.random()))[:3]*255}
    data["colour_border"] = np.asarray([min(i*0.8, 255) for i in data["colour"]])
    data["border_thickness"] = border_thickness
    body = world.CreateDynamicBody(position=pos, angle=angle, userData=data)
    body.CreateFixture(shape=shape, density=density,
                       friction=friction, restitution=restitution)
    return body

def createWorld(theta):
    '''
        Creates a whole instance of the world.
        theta decomposes to
        * impulse x and y
        * paddle.x amd y
        * paddle.angle
        * object.angle
        * object restitution
        * object position
    '''
    # Get variables
    imp = (theta[0], theta[1])
    paddle_pos = (theta[2], theta[3])
    paddle_angle = theta[4]
    obj_angle = theta[5]
    obj_k = theta[6]
    obj_pos = (theta[7], theta[8])
    obj_density = theta[9]
    parent_obj_k = theta[10]
    parent_obj_density = theta[11]
    # print theta

    # Construct world
    world = createBaseWorld()
    createStaticPaddle(world, paddle_pos, paddle_angle)

    # Create parent body
    createConvexBody(world, (-1, 6.5), 0, density = parent_obj_density,
                     restitution = parent_obj_k)

    body = createConvexBody(world, obj_pos, obj_angle, density = obj_density,
                            restitution = obj_k, border_thickness = 1)
    body.ApplyLinearImpulse(impulse=imp, point=body.worldCenter, wake=True)

    return world, body

def drawWorld(world):
    screen = np.ones((SCREEN_HEIGHT, SCREEN_WIDTH, 3), np.uint8) * \
                                                            BACKGROUND_COLOUR

    for body in world.bodies:
        for fixture in body.fixtures:
            fixture.shape.draw(screen, body, fixture)
    # cv2.imwrite("/data/test.png", screen)
    return screen

def drawContacts(screen, world, c):
    for contact_edge in c.contacts:
        if contact_edge.contact.touching:
            contact_edge.contact.draw(screen, c)
    return screen

def drawDist(screen, pointA, pointB):
    vertices = engine2canvas_vertices([np.asarray(pointA) * PPM,
                                       np.asarray(pointB) * PPM])
    cv2.line(screen, vertices[0], vertices[1], [128, 128, 40])
    return screen

def drawModelPredictions(screen, c, delta_p, old_outputs):
    # print '@@@@@@@@'
    # print 'c.worldCenter ', c.worldCenter
    # print 'old_outputs ', old_outputs
    # print 'vec(old_outputs) ', getList2b2Vec2(old_outputs)
    # print '@@@@@@@@'
    scale = 10
    vertices = engine2canvas_vertices([np.asarray(c.worldCenter) * PPM,
                                       np.asarray(c.worldCenter + scale * delta_p) * PPM,
                                       np.asarray(c.worldCenter + scale * getList2b2Vec2(old_outputs)) * PPM])
    cv2.line(screen, vertices[0], vertices[1], [128, 240, 40])
    cv2.line(screen, vertices[0], vertices[2], [240, 128, 40])

    cv2.putText(screen, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.8, [128, 240, 40])
    cv2.putText(screen, "Prediction", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
        0.8, [240, 128, 40])
    return screen

def getDeepConversion(b2X):
    return [b2X[0], b2X[1]]

def getList2b2Vec2(L2b):
    return Box2D.b2Vec2(L2b[0], L2b[1])

def b2Vec2Norm(b2v):
    return np.sqrt(b2v[0]**2 + b2v[1]**2)

def GetImpactV(world, c):
    for contact_edge in c.contacts:
        if contact_edge.contact.touching:
            vel1 = contact_edge.contact.fixtureA.body.GetLinearVelocityFromWorldPoint(contact_edge.contact.worldManifold.points[0])
            vel2 = contact_edge.contact.fixtureB.body.GetLinearVelocityFromWorldPoint(contact_edge.contact.worldManifold.points[0])
            impact_v = vel2 - vel1
            return impact_v
    return None # Box2D.b2Vec2(0, 0)


def getClosestDist(world, c):
    dists = []
    pointAs = []
    pointBs = []
    for body in world.bodies:
        if (body.mass > .0 and body != c):
            dist_result = b2Distance(shapeA=c.fixtures[0].shape,
                                     shapeB=body.fixtures[0].shape,
                                     transformA=c.transform,
                                     transformB=body.transform)
            pointA, pointB, dist, iterations = dist_result
            dists.append(dist)
            pointAs.append(pointA)
            pointBs.append(pointB)
    midx = dists.index(min(dists))
    return dists[midx], pointAs[midx], pointBs[midx]

def checkIfWorldStatic(world):
    isWorldStatic = True
    for body in world.bodies:
        if (not approxEqual(body.linearVelocity, Box2D.b2Vec2(0, 0), .01) or
            body.angularVelocity != 0.0):
            isWorldStatic = False
            break
    return isWorldStatic

def simulateWorld(theta, sim_length_s=5, saveVideo=False, filename="outputvideo2.mp4"):
    # print '----'
    world, c = createWorld(theta)

    if (saveVideo):
        # Save video
        writer = skvideo.io.FFmpegWriter(filename, outputdict={
          '-vcodec': 'libx264', '-b': '300000000', '-r': '60'})

    # Timestep of 1/60 of a second (60Hz) and 10 velocity/10 position iterations
    timeStep = 1.0 / 60
    vel_iters, pos_iters = 10, 10

    # Collect body information
    position = []
    orientation = []
    velocity = []
    impactVs = []
    distances2Obj = []
    collisions_count = 0

    # Main simulation loop
    for i in range(60 * sim_length_s):
        # Instruct the world to perform a single step of simulation.
        world.Step(timeStep, vel_iters, pos_iters)

        # Clear applied body forces. We didn't apply any forces, but you should
        # know about this function.
        # world.ClearForces()
        isWorldStatic = checkIfWorldStatic(world)


        # Get distance to closest other body
        mindist, pointA, pointB = getClosestDist(world, c)
        # Get impact v
        impV = GetImpactV(world, c)

        # Store data
        position.append(getDeepConversion(c.worldCenter))
        orientation.append(c.angle)
        velocity.append(getDeepConversion(c.linearVelocity))
        if (impV):
            impactVs.append(impV)
        distances2Obj.append(mindist)
        if inBodyCollision(c):
            collisions_count += 1

        if (saveVideo):
            # Draw the world
            canvas = drawWorld(world)
            canvas = drawContacts(canvas, world, c)
            canvas = drawDist(canvas, pointA, pointB)
            writer.writeFrame(canvas.astype('uint8'))
        if (isWorldStatic):
            # print 'World is static, stopping at ', i
            break

    # print 'Simulation done.'
    if (saveVideo):
        writer.close() # Close video writter object
    return np.asarray(position), \
           np.asarray(orientation), \
           np.asarray(velocity), \
           np.asarray(distances2Obj), \
           collisions_count, \
           np.asarray(impactVs)

def extractCrops(world, c):
    data = []
    canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), np.uint8)
    c.fixtures[0].shape.draw(canvas, c)
    v = engine2canvas_vertices([c.worldCenter * PPM])[0]
    id_crop = canvas[v[1] - CROP_H_SIZE:v[1] + CROP_H_SIZE,
                     v[0] - CROP_H_SIZE:v[0] + CROP_H_SIZE]
    id_crop[id_crop != 0] = FOREGROUND_COLOUR

    full_canvas = {}
    full_canvas['mass'] = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), np.float32)
    full_canvas['velx'] = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), np.float32)
    full_canvas['vely'] = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), np.float32)
    full_canvas['avel'] = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), np.float32)
    full_canvas['rest'] = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), np.float32)
    for body in world.bodies:
        canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), np.float32)
        mass = 1e5 # Default static mass in kg
        if (body.mass > .0):
            mass = body.mass
        for fixture in body.fixtures:
            fixture.shape.draw(canvas, body)
            canvas[canvas != 0] = FOREGROUND_COLOUR
            full_canvas['mass'] += canvas * mass
            full_canvas['velx'] += canvas * body.linearVelocity.x
            full_canvas['vely'] += canvas * body.linearVelocity.y
            full_canvas['avel'] += canvas * body.angularVelocity
            full_canvas['rest'] += canvas * fixture.restitution
                # print np.dstack((np.dstack((canvas, canvas)), canvas)).shape
                # print write_full_canvas.shape


    # print crop.shape
    data.append(id_crop)
    surround_crop = full_canvas['mass'][v[1] - CROP_H_SIZE:v[1] + CROP_H_SIZE,
                                        v[0] - CROP_H_SIZE:v[0] + CROP_H_SIZE]
    surround_crop[surround_crop != 0] = 255 # empty = 0, where there was mass = 255
    data.append(surround_crop)
    data.append(full_canvas['mass'][v[1] - CROP_H_SIZE:v[1] + CROP_H_SIZE,
                                    v[0] - CROP_H_SIZE:v[0] + CROP_H_SIZE])
    data.append(full_canvas['velx'][v[1] - CROP_H_SIZE:v[1] + CROP_H_SIZE,
                                    v[0] - CROP_H_SIZE:v[0] + CROP_H_SIZE])
    data.append(full_canvas['vely'][v[1] - CROP_H_SIZE:v[1] + CROP_H_SIZE,
                                    v[0] - CROP_H_SIZE:v[0] + CROP_H_SIZE])
    data.append(full_canvas['avel'][v[1] - CROP_H_SIZE:v[1] + CROP_H_SIZE,
                                    v[0] - CROP_H_SIZE:v[0] + CROP_H_SIZE])
    data.append(full_canvas['rest'][v[1] - CROP_H_SIZE:v[1] + CROP_H_SIZE,
                                    v[0] - CROP_H_SIZE:v[0] + CROP_H_SIZE])

    return data

def generateData(theta, sim_length_s=5):
    world, c = createWorld(theta)
    timeStep = 1.0 / 60
    vel_iters, pos_iters = 10, 10
    writer = skvideo.io.FFmpegWriter("/data/neuro_phys_sim/data/data.mp4", outputdict={
          '-vcodec': 'libx264', '-b': '300000000', '-r': '60'})

    data = {'cid':[], 'scrop': [], 'mass':[],
            'velx':[], 'vely':[],
            'avel':[], 'rest':[]}
    outdata = {'px':[], 'py':[], 'angle':[], 'vx':[], 'vy':[], 'vangle':[]}

    # Main simulation loop
    for i in range(60 * sim_length_s + 1): # To allow datasize to be exactly 60*s
        # Instruct the world to perform a single step of simulation.
        world.Step(timeStep, vel_iters, pos_iters)
        isWorldStatic = checkIfWorldStatic(world)

        crops = extractCrops(world, c)
        # print crop.shape
        data['cid'].append(crops[0])
        data['scrop'].append(crops[1])
        data['mass'].append(crops[2])
        data['velx'].append(crops[3])
        data['vely'].append(crops[4])
        data['avel'].append(crops[5])
        data['rest'].append(crops[6])

        outdata['px'].append(c.worldCenter.x)
        outdata['py'].append(c.worldCenter.y)
        outdata['angle'].append(c.angle)
        outdata['vx'].append(c.linearVelocity.x)
        outdata['vy'].append(c.linearVelocity.y)
        outdata['vangle'].append(c.angularVelocity)

        # cv2.imwrite("data/cid.png", np.asarray(data['cid'][-1]*255))
        # cv2.imwrite("data/scrop.png", np.asarray(data['scrop'][-1]*255))
        # cv2.imwrite("data/mass.png", np.asarray(data['mass'][-1]*255))
        # cv2.imwrite("data/velx.png", np.asarray(data['velx'][-1]*255))
        # cv2.imwrite("data/vely.png", np.asarray(data['vely'][-1]*255))
        # cv2.imwrite("data/avel.png", np.asarray(data['avel'][-1]*255))
        # cv2.imwrite("data/rest.png", np.asarray(data['rest'][-1]*255))
        # # print data
        # if (i >10):
        #     break

        write_full_canvas = np.dstack((np.dstack((crops[1], crops[1])), crops[1]))
        writer.writeFrame(write_full_canvas.astype('uint8'))

        # Stop if world is no longer moving
        if (isWorldStatic):
            # print 'World is static, stopping at ', i
            break

    # remove last element as it doesn't have a corresponding outdata element
    data['cid'].pop()
    data['scrop'].pop()
    data['mass'].pop()
    data['velx'].pop()
    data['vely'].pop()
    data['avel'].pop()
    data['rest'].pop()

    # cv2.imwrite("data/cid.png", np.asarray(data['cid'][-1]*255))
    # cv2.imwrite("data/scrop.png", np.asarray(data['scrop'][-1]*255))
    # cv2.imwrite("data/mass.png", np.asarray(data['mass'][-1]*255))
    # cv2.imwrite("data/velx.png", np.asarray(data['velx'][-1]*255))
    # cv2.imwrite("data/vely.png", np.asarray(data['vely'][-1]*255))
    # cv2.imwrite("data/avel.png", np.asarray(data['avel'][-1]*255))
    # cv2.imwrite("data/rest.png", np.asarray(data['rest'][-1]*255))

    # Calculate the delta positions and shift the rest of the data by one to
    # correspond to the data samples
    for key in ['px', 'py', 'angle']:
        outdata[key] = [outdata[key][i-1] - outdata[key][i] for i in xrange(1, len(outdata[key]))]
    for key in ['vx', 'vy', 'vangle']:
        outdata[key].pop(0) # remove first element. we need the V in the next timestep
    # print np.asarray(data['mass']).shape
    # print np.asarray(outdata['px']).shape

    writer.close()
    return data, outdata

def simulateWithModel(theta, model_func, sim_length_s=5, threshold_sigma=3.0):
    world, c = createWorld(theta)
    timeStep = 1.0 / 60
    vel_iters, pos_iters = 10, 10
    writer = skvideo.io.FFmpegWriter("/data/neuro_phys_sim/data/model_eval.mp4", outputdict={
          '-vcodec': 'libx264', '-b': '300000000', '-r': '60'})

    predicted_output = []
    real_output = []
    sigma_prob = []

    wrong_pred_crops = []
    wrong_pred_output = []

    old_outputs = [0, 0]
    old_variances = [1e-5, 1e-5]
    old_pos = c.worldCenter
    old_angle = c.angle
    old_crops = []
    for i in range(60 * sim_length_s + 1): # To allow datasize to be exactly 60*s
        world.Step(timeStep, vel_iters, pos_iters)
        isWorldStatic = checkIfWorldStatic(world)

        crops = extractCrops(world, c)
        ncrops = [np.asarray(crops[ci]).reshape(1, 32, 32, 1) for ci in xrange(len(crops))]

        new_outputs, new_variances = model_func(ncrops)
        new_variances = np.exp(np.asarray(new_variances, dtype='float64').flatten()/2.0)

        new_outputs = np.asarray(new_outputs, dtype='float64').flatten()

        delta_p = old_pos -  c.worldCenter

        # Get probability
        from numpy.linalg import inv
        sprob = np.abs(np.matmul(inv(np.diag(old_variances)),
            np.asarray(getDeepConversion(delta_p - old_outputs))))
        sigma_prob.append(sprob)
        # print 'Sprob: ', sprob
        if (np.max(sprob) > threshold_sigma):
            # print 'FAR TOOO LARGE!! '
            # Get wrong samples for new training data!
            if (old_crops):
                wrong_pred_crops.append(old_crops)
                wrong_pred_output.append([delta_p.x, delta_p.y, old_angle - c.angle,
                         c.linearVelocity.x, c.linearVelocity.y, c.angularVelocity])
        else:
            print 'I\'m within bounds! ', sprob

        if (b2Vec2Norm(delta_p - getList2b2Vec2(old_outputs)) / b2Vec2Norm(delta_p) > 0.1):
            impV = GetImpactV(world, c)
            # print 'Impact velocity ', impV
            # print 'THINGS @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'
            ## Get object data
            ##

        # Store outputs
        predicted_output.append(old_outputs)
        real_output.append(delta_p)

        # Draw the world
        canvas = drawWorld(world)
        canvas = drawModelPredictions(canvas, c, delta_p, old_outputs)
        writer.writeFrame(canvas.astype('uint8'))

        old_outputs = new_outputs
        old_variances = new_variances
        old_pos = getList2b2Vec2(c.worldCenter)
        old_angle = c.angle
        old_crops = crops
        # Stop if world is no longer moving
        if (isWorldStatic):
            print 'World is static, stopping at ', i
            break
    writer.close()
    real_output = [[x[0], x[1]] for x in real_output]
    return real_output, predicted_output, sigma_prob, wrong_pred_crops, wrong_pred_output

def updateArchiveDirectly(archive, data, outdata):
    data_length = len(data)
    print 'updateArchiveDirectly ', len(data), len(outdata), data_length
    with h5py.File(archive, 'a') as f: # Read/write if exists, create otherwise
        data_sh = (0, 7, CROP_SIZE, CROP_SIZE)
        if 'data' in f:
            data_sh = f['data'].shape

        data_set = f.require_dataset('data', shape = data_sh,
            maxshape=(None, 7, CROP_SIZE, CROP_SIZE), dtype='float32',
            compression="gzip", compression_opts=9)

        outdata_sh = (0, 6)
        if 'outdata' in f:
            outdata_sh = f['outdata'].shape

        outdata_set = f.require_dataset('outdata', shape = outdata_sh,
            maxshape=(None, 6), dtype='float32',
            compression="gzip", compression_opts=5)

        # Update data at the same time
        data_set.resize(data_set.shape[0] + data_length, axis=0)
        data_set[-data_length:] = data
        outdata_set.resize(outdata_set.shape[0] + data_length, axis=0)
        outdata_set[-data_length:] = outdata

        return data_set.shape[0], outdata_set.shape[0]

def updateArchive(archive, data, outdata):
    # print 'TO TEST and create a unit test'
    data_length = len(data['cid'])

    ndata = np.empty(shape=(data_length, 0, CROP_SIZE, CROP_SIZE), dtype='float32')
    for k in ['cid', 'scrop', 'mass', 'velx', 'vely', 'avel', 'rest']:
        ndata = np.concatenate((ndata, np.expand_dims(np.asarray(data[k]), axis=1)), axis=1)

    noutdata = np.empty(shape=(data_length, 0), dtype='float32')
    for k in ['px', 'py', 'angle', 'vx', 'vy', 'vangle']:
        noutdata = np.concatenate((noutdata, np.expand_dims(np.asarray(outdata[k]), axis=1)), axis=1)

    return updateArchiveDirectly(archive, ndata, noutdata)


def getDataFromArchive(filename, sample_from_data=False):
    with h5py.File(filename, 'a') as f:
        if (sample_from_data):
            return f['data'][:256*8], f['outdata'][:256*8]
        else:
            return f['data'][()], f['outdata'][()]

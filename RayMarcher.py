import taichi as ti 
import taichi.math as math
import random
import math

ti.init(arch=ti.gpu)
n=300 # resolution
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(n,n)) #Stores values per pixel to be displayed
screenSpace = ti.Vector.field(3, dtype=ti.f32, shape=(n,n)) #Position vector for each pixel
objectNormal = ti.Vector.field(3, dtype=ti.f32, shape=(n,n))
color = ti.Vector([.4,1,1]) #random color vector

rayDir = ti.Vector.field(3, dtype=ti.f32, shape=(n,n)) #Direction of ray
t = ti.field(dtype=ti.f32, shape=(n,n)) #Total Distance to travel
d = ti.field(dtype=ti.f32, shape=(n,n)) #Distance ray is from SDF
p = ti.Vector.field(3, dtype=ti.f32, shape=(n,n)) #Current position of ray
dMin = ti.field(dtype=int, shape=(n,n))

MaxDistance = 40
rayOrigin = ti.Vector([0,0,-3]) #Starting point of ray
Intensity = 1
fov = 1

diffuseStrength = ti.field(dtype=ti.f32, shape=(n,n))
specularStrength = ti.field(dtype=ti.f32, shape=(n,n))

#SDF Objects
#Sphere = ti.Vector([0,0,1]) #Origin vector

@ti.func
def Rodrig(p, a, angle):
    return ti.math.mix(ti.math.dot(a,p) * a, p, ti.math.cos(angle)) + ti.math.cross(a,p) * ti.math.sin(a)

@ti.func
def CalcSceneNormal(p, Objects): #Calculates normals of generated objects, using small differentials
    d = ti.Vector([0.001, 0])
    gx = SDF(p + ti.Vector([d[0], d[1], d[1]]), Objects) - SDF(p - ti.Vector([d[0], d[1], d[1]]), Objects)
    gy = SDF(p + ti.Vector([d[1], d[0], d[1]]), Objects) - SDF(p - ti.Vector([d[1], d[0], d[1]]), Objects)
    gz = SDF(p + ti.Vector([d[1], d[1], d[0]]), Objects) - SDF(p - ti.Vector([d[1], d[1], d[0]]), Objects)
    normal = ti.Vector.normalized(ti.Vector([gx, gy, gz]))
    return normal

@ti.func
def SDF(point, Objects):
    qoint = ti.math.fract(point) - .5 #Repeating
    roint = Rodrig(point, ti.Vector([0,0,point[2]]), ti.math.pi) #Rotated

    Sphere1 = ti.math.length(point - Objects[0]) - .2
    Sphere2 = ti.math.length(point - Objects[1]) - .5

    ground = (point[1]) +.75 + ti.math.sin(point[2]*10)*.1
    d = ti.math.min(Sphere1, Sphere2, ground)
    return d

@ti.func
def Scene(a):
    Sphere1 = ti.Vector([0,0,-.1])
    Sphere2 = ti.Vector([ti.math.sin(a),ti.math.cos(a),-.1])
    return [Sphere1, Sphere2]

@ti.func
def RaymarchFunc(a, p, rd, dist):
    Object = Scene(a)
    total = 1.7
    marchDist = 0
    marchDistMin = 0
    for i in range(50):
        if not(marchDistMin):
            s = p + rd * total
            marchDist = SDF(s, Object)
            total += marchDist
            if total > dist:
                marchDistMin = 1
    return total

@ti.func
def Shadowmarch(a, p, rd, dist):
    Object = Scene(a)
    total = 0
    marchDist = 0
    marchDistMin = 0
    s = p + rd * total
    marchDist = SDF(s, Object)
    total += marchDist
    return total


@ti.kernel
def paint(a:ti.f32):
    Object = Scene(a)
    lightColor = ti.Vector([1,1,1]) 
    lightSource = ti.Vector([1,2,-1]) * 1
    lightDirection = ti.Vector.normalized(lightSource)
    for i, j in pixels: #parallized over pixels
        if t[i,j] < MaxDistance-5:
            objectNormal[i,j] = CalcSceneNormal(p[i,j], Object)
        else:
            objectNormal[i,j] = ti.Vector([0,0,0])

        diffuseStrength[i,j] = ti.math.max(0.0, ti.math.dot(ti.Vector.normalized(lightSource), objectNormal[i,j]))
        viewSource = ti.Vector.normalized(rayDir[i,j])
        reflectSource = ti.Vector.normalized(ti.math.reflect(lightSource, objectNormal[i,j]))
        specularStrength[i,j] = ti.math.pow(ti.math.max(0.0, ti.math.dot(viewSource,reflectSource)), 64)
        pixels[i,j] = ti.math.pow(lightColor * diffuseStrength[i,j] *.15 + lightColor * specularStrength[i,j] *.85,ti.Vector([1/2.2,1/2.2,1/2.2]))

        rayOriginProx = p[i,j] + objectNormal[i,j] * .1
        distToLight = ti.math.distance(rayOriginProx, lightSource)
        shadowDist = RaymarchFunc(a, rayOriginProx, lightDirection, distToLight)
        if shadowDist < distToLight:
            pixels[i,j] = pixels[i,j] * ti.Vector([.25, .25, .25])
        #pixels[i,j] = objectNormal[i,j]
        t[i,j] = 0
        dMin[i,j] = 0

@ti.kernel
def Raymarch(a:ti.f32):
    Object = Scene(a)
    for i, j in pixels:
        if not(dMin[i,j]):
            p[i,j] = rayOrigin + rayDir[i,j] * t[i,j]
            d[i,j] = SDF(p[i,j], Object)
            t[i,j] += d[i,j]
            if t[i,j] > MaxDistance:
                t[i,j] = MaxDistance
            if d[i,j] < .005 or t[i,j] > MaxDistance:
                dMin[i,j] = 1
        #objectNormal[i,j] = CalcSceneNormal(p[i,j], Object, t[i,j])

@ti.kernel
def setup():
    #CalcScreenSpace()
    for i, j in pixels:
        screenSpace[i,j] = ti.Vector([(i*2 - pixels.shape[0])/pixels.shape[0], (j*2 - pixels.shape[1])/pixels.shape[1], 0])
        rayDir[i,j][0] = screenSpace[i,j][0] * fov
        rayDir[i,j][1] = screenSpace[i,j][1] * fov
        rayDir[i,j][2] = 1
        rayDir[i,j] = ti.Vector.normalized(rayDir[i,j])
        t[i,j] = 0
        d[i,j] = 0


gui = ti.GUI("Cell Auto", res=(n,n))
setup()
it = 0
while gui.running:
    it += .01
    for i in range(50):
        Raymarch(it)
    paint(it)
    gui.set_image(pixels)
    gui.show()
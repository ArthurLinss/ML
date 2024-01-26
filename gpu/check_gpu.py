import objc
#from pyobjc import *
from Quartz import *

def get_gpu_info():
    devices = []
    service_dict = IOServiceMatching("IOMobileFramebuffer")
    iterator = IOServiceGetMatchingServices(kIOMasterPortDefault, service_dict, None)
    while True:
        framebuffer = IOIteratorNext(iterator)
        if not framebuffer:
            break
        framebuffer_info = IORegistryEntryCreateCFProperty(framebuffer, "IOFramebuffer", kCFAllocatorDefault, 0)
        devices.append(framebuffer_info)
        IOObjectRelease(framebuffer)

    IOObjectRelease(iterator)
    return devices

gpu_info = get_gpu_info()
print(gpu_info)

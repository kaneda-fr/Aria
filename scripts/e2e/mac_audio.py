"""Read microphone consent for this Python process; never modifies macOS permissions."""
import ctypes


def microphone_authorization() -> str:
    ctypes.CDLL("/System/Library/Frameworks/AVFoundation.framework/AVFoundation")
    objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    make_string = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p)(
        ("objc_msgSend", objc))
    audio = make_string(objc.objc_getClass(b"NSString"),
                        objc.sel_registerName(b"stringWithUTF8String:"), b"soun")
    get_status = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
        ("objc_msgSend", objc))
    status = get_status(objc.objc_getClass(b"AVCaptureDevice"),
                        objc.sel_registerName(b"authorizationStatusForMediaType:"), audio)
    return {0: "not determined", 1: "restricted", 2: "denied", 3: "authorized"}.get(status, f"unknown ({status})")


if __name__ == "__main__":
    print(microphone_authorization())

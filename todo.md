
### separate module for basic image operation
  
  - scale
  - rotate
  - crop
  - flip

### support for ESP32 Cam

default config (1600x1200): 
  - port: 8080
    mode: stream
  - port: 8081
    mode: snapshot

i.e. `curl -o snapshot.jpeg http://192.168.1.224:8081` just give a current snapshot


### probably use STB image

```
// Define the compilation flags
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
```
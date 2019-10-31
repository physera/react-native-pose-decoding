# react-native-pose-decoding

## Getting started

`$ npm install react-native-pose-decoding --save`

### Mostly automatic installation

`$ react-native link react-native-pose-decoding`

### Manual installation


#### Android

1. Open up `android/app/src/main/java/[...]/MainApplication.java`
  - Add `import com.indigoviolet.posedecoding.RNPoseDecodingPackage;` to the imports at the top of the file
  - Add `new RNPoseDecodingPackage()` to the list returned by the `getPackages()` method
2. Append the following lines to `android/settings.gradle`:
  	```
  	include ':react-native-pose-decoding'
  	project(':react-native-pose-decoding').projectDir = new File(rootProject.projectDir, 	'../node_modules/react-native-pose-decoding/android')
  	```
3. Insert the following lines inside the dependencies block in `android/app/build.gradle`:
  	```
      compile project(':react-native-pose-decoding')
  	```


## Usage
```javascript
import RNPoseDecoding from 'react-native-pose-decoding';

// TODO: What to do with the module?
RNPoseDecoding;
```

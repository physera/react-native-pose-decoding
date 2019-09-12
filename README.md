# react-native-posenet

## Getting started

`$ npm install react-native-posenet --save`

### Mostly automatic installation

`$ react-native link react-native-posenet`

### Manual installation


#### Android

1. Open up `android/app/src/main/java/[...]/MainApplication.java`
  - Add `import com.indigoviolet.posenet.RNPosenetPackage;` to the imports at the top of the file
  - Add `new RNPosenetPackage()` to the list returned by the `getPackages()` method
2. Append the following lines to `android/settings.gradle`:
  	```
  	include ':react-native-posenet'
  	project(':react-native-posenet').projectDir = new File(rootProject.projectDir, 	'../node_modules/react-native-posenet/android')
  	```
3. Insert the following lines inside the dependencies block in `android/app/build.gradle`:
  	```
      compile project(':react-native-posenet')
  	```


## Usage
```javascript
import RNPosenet from 'react-native-posenet';

// TODO: What to do with the module?
RNPosenet;
```
  
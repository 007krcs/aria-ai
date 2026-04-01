import { CapacitorConfig } from "@capacitor/cli";

const config: CapacitorConfig = {
  appId:    "com.aria.personal",
  appName:  "ARIA",
  webDir:   "dist",
  server: {
    // During development, point to the live ARIA backend
    // Comment this out for production APK/IPA builds
    // url: "http://192.168.1.x:8000",
    cleartext: true,
  },
  android: {
    buildOptions: {
      releaseType: "APK",
    },
    backgroundColor: "#00050f",
  },
  ios: {
    contentInset: "always",
    backgroundColor: "#00050f",
  },
  plugins: {
    SplashScreen: {
      launchShowDuration: 2000,
      backgroundColor: "#00050f",
      androidSplashResourceName: "splash",
      androidScaleType: "CENTER_CROP",
      showSpinner: false,
    },
  },
};

export default config;

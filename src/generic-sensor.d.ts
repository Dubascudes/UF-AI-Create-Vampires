// Generic Sensor API orientation sensor (https://w3c.github.io/orientation-sensor/), shipped in Chrome for Android.
interface OrientationSensorOptions {
  frequency?: number;
  referenceFrame?: 'device' | 'screen';
}

declare class RelativeOrientationSensor extends EventTarget {
  constructor(options?: OrientationSensorOptions);
  readonly quaternion?: number[] | null;
  readonly activated: boolean;
  onerror: ((event: Event & { error: DOMException }) => void) | null;
  start(): void;
  stop(): void;
}

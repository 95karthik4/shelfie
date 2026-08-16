/**
 * In-app capture. The primary way into the app: the brief is "takes or picks
 * a photo", and a picker-only app only does half of that.
 */

import { CameraView, useCameraPermissions } from 'expo-camera';
import React, { useRef, useState } from 'react';
import { SafeAreaView, StyleSheet, Text, View } from 'react-native';

import { colors, spacing } from '../theme';
import { Button } from '../components/ui';

interface Props {
  onCaptured: (uri: string) => void;
  onCancel: () => void;
}

export function CameraScreen({ onCaptured, onCancel }: Props) {
  const [permission, requestPermission] = useCameraPermissions();
  const [busy, setBusy] = useState(false);
  const cameraRef = useRef<CameraView | null>(null);

  // Permission is still being read from the OS.
  if (!permission) {
    return (
      <SafeAreaView style={styles.centered}>
        <Text style={styles.message}>Checking camera permission…</Text>
      </SafeAreaView>
    );
  }

  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.centered}>
        <Text style={styles.heading}>Camera access needed</Text>
        <Text style={styles.message}>
          Shelfie needs the camera to photograph your bookshelf. You can also pick an existing
          photo from your library instead.
        </Text>
        <Button label="Allow camera" onPress={() => void requestPermission()} style={styles.button} />
        <Button label="Back" variant="secondary" onPress={onCancel} style={styles.button} />
      </SafeAreaView>
    );
  }

  async function capture() {
    if (!cameraRef.current || busy) {
      return;
    }
    setBusy(true);
    try {
      const photo = await cameraRef.current.takePictureAsync({ quality: 0.9 });
      if (photo?.uri) {
        onCaptured(photo.uri);
      }
    } catch {
      // Capture failing is rare and recoverable: stay on the camera so the
      // user can simply try again.
      setBusy(false);
      return;
    }
    setBusy(false);
  }

  return (
    <View style={styles.container}>
      <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing="back" />
      <SafeAreaView style={styles.overlay}>
        <View style={styles.hintBox}>
          <Text style={styles.hint}>Fill the frame with one shelf. Hold steady.</Text>
        </View>
        <View style={styles.controls}>
          <Button label="Cancel" variant="secondary" onPress={onCancel} style={styles.cancel} />
          <Button label="Capture" onPress={() => void capture()} busy={busy} style={styles.shutter} />
        </View>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  overlay: { flex: 1, justifyContent: 'space-between' },
  hintBox: {
    margin: spacing.lg,
    backgroundColor: 'rgba(0,0,0,0.55)',
    borderRadius: 10,
    padding: spacing.md,
  },
  hint: { color: '#FFF', textAlign: 'center', fontSize: 14 },
  controls: {
    flexDirection: 'row',
    gap: spacing.md,
    padding: spacing.lg,
  },
  cancel: { flex: 1 },
  shutter: { flex: 2 },
  centered: {
    flex: 1,
    backgroundColor: colors.background,
    justifyContent: 'center',
    padding: spacing.xl,
  },
  heading: { fontSize: 22, fontWeight: '700', color: colors.text, marginBottom: spacing.sm },
  message: { fontSize: 15, color: colors.textMuted, lineHeight: 22 },
  button: { marginTop: spacing.md },
});

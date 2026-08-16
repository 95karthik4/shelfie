/**
 * Shelfie root. The whole flow is one state machine and two tabs -- no
 * navigation library, because there are five screens and one linear path
 * through them:
 *
 *   home -> camera/picker -> preview -> analyzing -> results
 *
 * Screen state lives here so the scan survives tab switches: a user can check
 * the library mid-review and come back to the same results.
 */

import { StatusBar } from 'expo-status-bar';
import React, { useState } from 'react';
import { SafeAreaView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';

import { ApiError, ScanResponse, uploadScan } from './src/api';
import { Decision, DecisionMap } from './src/decisions';
import { colors, spacing } from './src/theme';
import { AnalyzingScreen } from './src/screens/AnalyzingScreen';
import { CameraScreen } from './src/screens/CameraScreen';
import { HomeScreen } from './src/screens/HomeScreen';
import { LibraryScreen } from './src/screens/LibraryScreen';
import { PreviewScreen } from './src/screens/PreviewScreen';
import { ResultsScreen } from './src/screens/ResultsScreen';

type Phase = 'home' | 'camera' | 'preview' | 'analyzing' | 'results';
type Tab = 'scan' | 'library';

export default function App() {
  const [tab, setTab] = useState<Tab>('scan');
  const [phase, setPhase] = useState<Phase>('home');
  const [photoUri, setPhotoUri] = useState<string | null>(null);
  const [scan, setScan] = useState<ScanResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Bumped on every confirmation so the library refetches when shown.
  const [libraryVersion, setLibraryVersion] = useState(0);
  // Per-item decisions live here, not in the cards: switching to the Library
  // tab unmounts the results tree, and a decision must not die with it.
  const [decisions, setDecisions] = useState<DecisionMap>({});

  function startOver() {
    setPhase('home');
    setPhotoUri(null);
    setScan(null);
    setError(null);
    setDecisions({});
  }

  function recordDecision(itemId: number, decision: Decision | null) {
    setDecisions((current) => {
      if (decision === null) {
        // Undo -- drop the entry so the item is undecided again.
        const { [itemId]: _removed, ...rest } = current;
        return rest;
      }
      return { ...current, [itemId]: decision };
    });
    // Only a confirmation changes what the server holds.
    if (decision?.kind === 'confirmed') {
      setLibraryVersion((version) => version + 1);
    }
  }

  async function analyze() {
    if (!photoUri) {
      return;
    }
    setPhase('analyzing');
    setError(null);
    try {
      const result = await uploadScan(photoUri);
      setScan(result);
      setPhase('results');
    } catch (caught) {
      // Back to preview, not to home: the photo is still good, and the user
      // should be able to retry without recapturing.
      setError(
        caught instanceof ApiError ? caught.message : 'The scan failed for an unknown reason.'
      );
      setPhase('preview');
    }
  }

  function renderScan() {
    if (phase === 'camera') {
      return (
        <CameraScreen
          onCaptured={(uri) => {
            setPhotoUri(uri);
            setError(null);
            setPhase('preview');
          }}
          onCancel={() => setPhase('home')}
        />
      );
    }
    if (phase === 'preview' && photoUri) {
      return (
        <PreviewScreen
          uri={photoUri}
          error={error}
          onRetake={startOver}
          onAnalyze={() => void analyze()}
        />
      );
    }
    if (phase === 'analyzing' && photoUri) {
      return <AnalyzingScreen uri={photoUri} />;
    }
    if (phase === 'results' && scan) {
      return (
        <ResultsScreen
          scan={scan}
          decisions={decisions}
          onDecision={recordDecision}
          onDone={() => {
            startOver();
            setTab('library');
          }}
          onScanAnother={startOver}
        />
      );
    }
    return (
      <HomeScreen
        onOpenCamera={() => setPhase('camera')}
        onPicked={(uri) => {
          setPhotoUri(uri);
          setError(null);
          setPhase('preview');
        }}
      />
    );
  }

  // The camera is full-bleed; every other screen sits inside the safe area.
  const fullBleed = tab === 'scan' && phase === 'camera';

  return (
    <View style={styles.root}>
      <StatusBar style={fullBleed ? 'light' : 'dark'} />
      {fullBleed ? (
        renderScan()
      ) : (
        <SafeAreaView style={styles.safe}>
          <View style={styles.content}>
            {tab === 'scan' ? renderScan() : <LibraryScreen reloadKey={libraryVersion} />}
          </View>
          {phase === 'analyzing' ? null : (
            <View style={styles.tabs}>
              <TabButton label="Scan" active={tab === 'scan'} onPress={() => setTab('scan')} />
              <TabButton
                label="Library"
                active={tab === 'library'}
                onPress={() => setTab('library')}
              />
            </View>
          )}
        </SafeAreaView>
      )}
    </View>
  );
}

function TabButton({
  label,
  active,
  onPress,
}: {
  label: string;
  active: boolean;
  onPress: () => void;
}) {
  return (
    <TouchableOpacity
      accessibilityRole="tab"
      accessibilityState={{ selected: active }}
      style={[styles.tab, active && styles.tabActive]}
      onPress={onPress}
      activeOpacity={0.7}
    >
      <Text style={[styles.tabLabel, active && styles.tabLabelActive]}>{label}</Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.background },
  safe: { flex: 1 },
  content: { flex: 1 },
  tabs: {
    flexDirection: 'row',
    borderTopWidth: 1,
    borderTopColor: colors.border,
    backgroundColor: colors.surface,
  },
  tab: { flex: 1, paddingVertical: spacing.md, alignItems: 'center' },
  tabActive: { borderTopWidth: 2, borderTopColor: colors.primary },
  tabLabel: { fontSize: 15, fontWeight: '600', color: colors.textMuted },
  tabLabelActive: { color: colors.primary },
});

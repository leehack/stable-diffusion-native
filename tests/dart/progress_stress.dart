// Regression harness for progress reporting while the Dart VM shuts down.
//
// Usage: dart progress_stress.dart <library> <design> <owner> <reportMs>
//            <gapUs> <dieAfterMs> <allocatingIsolates> [lossless]
//
// A worker isolate blocks in a native call that reports progress, as
// generate_image() does. Other isolates allocate, so the isolate group keeps
// requesting safepoints. The isolate that consumes progress (the "owner")
// then goes away while the worker is still reporting:
//
//   owner main        the main isolate owns progress and dies of an error
//   owner background  a background isolate owns it; the main isolate dies
//   owner killed      a background isolate owns it and is killed; then the
//                     main isolate dies
//
// How the owner consumes progress:
//
//   design poll    sd_dart_progress_enable() and a timer that reads
//                  sd_dart_progress_read(); no Dart callback exists. Every
//                  report read must be the one its sequence names, in
//                  order, and reports may be missing only when more were
//                  recorded than the library keeps; with `lossless`, never
//   design raw     control: a NativeCallable.listener installed with
//                  sd_set_progress_callback()
//   design locked  control: the listener behind a forwarder that locks
//                  around the call, cleared by a NativeFinalizer
//
// The process must end with the unhandled error (exit code 255) once the
// native call returns. The controls instead abort in the VM ("Callback invoked
// after it has been deleted", "GetFfiCallbackMetadata called after shutdown")
// or never exit.
import 'dart:async';
import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';

typedef ProgressNative = Void Function(Int, Int, Float, Pointer<Void>);
typedef SetNative = Void Function(
  Pointer<NativeFunction<ProgressNative>>,
  Pointer<Void>,
);
typedef SetDart = void Function(
  Pointer<NativeFunction<ProgressNative>>,
  Pointer<Void>,
);

final class Progress extends Struct {
  @Uint64()
  external int sequence;
  @Int32()
  external int step;
  @Int32()
  external int steps;
  @Float()
  external double time;
}

final class Anchor implements Finalizable {}

// Kept reachable: a NativeFinalizer that was collected does not run when its
// isolate shuts down.
final anchor = Anchor();
NativeFinalizer? keptFinalizer;
NativeCallable<ProgressNative>? keptCallback;
Timer? keptTimer;

// The number of recent reports sd_dart_wrapper.h promises to keep.
const int reportsKept = 4095;
const int batchSize = 256;

int progressSeen = 0;
int progressMissed = 0;

void fail(int code, String message) {
  stderr.writeln('stress: $message');
  exit(code);
}

void consumeProgress(String libraryPath, String design, bool lossless) {
  final library = DynamicLibrary.open(libraryPath);
  if (design == 'poll') {
    library.lookupFunction<Void Function(), void Function()>(
      'sd_dart_progress_enable',
    )();
    final read = library
        .lookupFunction<
          Size Function(Uint64, Pointer<Progress>, Size, Pointer<Uint64>),
          int Function(int, Pointer<Progress>, int, Pointer<Uint64>)
        >('sd_dart_progress_read', isLeaf: true);
    final malloc = DynamicLibrary.process()
        .lookupFunction<
          Pointer<Void> Function(IntPtr),
          Pointer<Void> Function(int)
        >('malloc');
    final reports = malloc(batchSize * sizeOf<Progress>()).cast<Progress>();
    final latest = malloc(sizeOf<Uint64>()).cast<Uint64>();
    var after = 0;
    keptTimer = Timer.periodic(const Duration(milliseconds: 1), (_) {
      for (var call = 0; call < 64; call++) {
        final count = read(after, reports, batchSize, latest);
        for (var i = 0; i < count; i++) {
          final report = reports[i];
          // The worker is the only reporter: its report n has sequence n + 1.
          final step = (report.sequence - 1) & 0x3fffffff;
          if (report.sequence != reports[0].sequence + i ||
              report.step != step ||
              report.steps != step + 1 ||
              report.time != step % 1024) {
            fail(
              4,
              'report ${report.sequence} is ${report.step} ${report.steps} '
              '${report.time}, at $i of a read after $after',
            );
          }
        }
        if (count > 0) {
          final missed = reports[0].sequence - after - 1;
          if (missed > 0 && (lossless || latest.value - after <= reportsKept)) {
            fail(
              6,
              '$missed reports after $after are missing with ${latest.value} '
              'recorded',
            );
          }
          progressMissed += missed;
          progressSeen += count;
          after = reports[count - 1].sequence;
        }
        if (after >= latest.value) {
          break;
        }
      }
    });
    return;
  }

  final callback = NativeCallable<ProgressNative>.listener((
    int step,
    int steps,
    double time,
    Pointer<Void> data,
  ) {
    progressSeen++;
  });
  keptCallback = callback;
  if (design == 'raw') {
    library.lookupFunction<SetNative, SetDart>('sd_set_progress_callback')(
      callback.nativeFunction,
      nullptr,
    );
  } else {
    keptFinalizer = NativeFinalizer(
      library.lookup<NativeFinalizerFunction>(
        'control_locked_clear_progress_callback',
      ),
    )..attach(anchor, callback.nativeFunction.cast());
    library.lookupFunction<SetNative, SetDart>(
      'control_locked_set_progress_callback',
    )(callback.nativeFunction, nullptr);
  }
}

void owner((SendPort, String, String, bool) arguments) {
  final (ready, libraryPath, design, lossless) = arguments;
  consumeProgress(libraryPath, design, lossless);
  final commands = ReceivePort();
  commands.listen((_) => ready.send((progressSeen, progressMissed)));
  ready.send(commands.sendPort);
}

void worker((SendPort, String, int, int) arguments) {
  final (started, libraryPath, reportMs, gapUs) = arguments;
  final report = DynamicLibrary.open(libraryPath)
      .lookupFunction<Long Function(Int, Int), int Function(int, int)>(
        'stress_report_progress',
      );
  started.send('reporting');
  report(reportMs, gapUs);
  started.send('reported');
}

void allocate(int seed) {
  var kept = <List<int>>[];
  var value = seed;
  while (true) {
    kept.add(List<int>.filled(256, value++));
    if (kept.length > 2000) {
      kept = <List<int>>[];
    }
  }
}

Future<void> main(List<String> arguments) async {
  final libraryPath = arguments[0];
  final design = arguments[1];
  final ownerLayout = arguments[2];
  final reportMs = int.parse(arguments[3]);
  final gapUs = int.parse(arguments[4]);
  final dieAfter = Duration(milliseconds: int.parse(arguments[5]));
  final allocatingIsolates = int.parse(arguments[6]);
  final lossless = arguments.length > 7 && arguments[7] == 'lossless';

  final messages = ReceivePort();
  final inbox = StreamIterator<Object?>(messages);
  Isolate? ownerIsolate;
  SendPort? ownerCommands;
  if (ownerLayout == 'main') {
    consumeProgress(libraryPath, design, lossless);
  } else {
    ownerIsolate = await Isolate.spawn(owner, (
      messages.sendPort,
      libraryPath,
      design,
      lossless,
    ));
    await inbox.moveNext();
    ownerCommands = inbox.current as SendPort;
  }
  for (var i = 0; i < allocatingIsolates; i++) {
    await Isolate.spawn(allocate, i);
  }
  await Isolate.spawn(worker, (
    messages.sendPort,
    libraryPath,
    reportMs,
    gapUs,
  ));
  await inbox.moveNext();

  await Future<void>.delayed(dieAfter);
  var (seen, missed) = (progressSeen, progressMissed);
  if (ownerCommands != null) {
    ownerCommands.send(null);
    await inbox.moveNext();
    if (inbox.current == 'reported') {
      fail(3, 'the native call returned before the owner died');
    }
    (seen, missed) = inbox.current as (int, int);
  }
  if (seen == 0) {
    fail(5, 'the owner saw no progress');
  }
  if (ownerLayout == 'killed') {
    final exited = ReceivePort();
    ownerIsolate!.addOnExitListener(exited.sendPort);
    ownerIsolate.kill(priority: Isolate.immediate);
    await exited.first;
  }
  stderr.writeln('stress: dying after $seen reports read, $missed missed');
  throw StateError('stress: unhandled error');
}

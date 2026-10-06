// Checks that a consumer that polls gets every progress report of a batch.
//
// Usage: dart progress_images.dart <library> <consumer> <images> <steps>
//            <stepUs> <tiles> <tileUs> <pollMs>
//
// stable-diffusion.cpp samples every image of a batch before it decodes any,
// so the last report of one image (`steps/steps`) is followed within a
// millisecond by the first of the next (`0/steps`). A consumer counts images
// by their last reports, so it must not miss one between two polls.
//
// A worker isolate runs a stand-in for generate_image() that reports as
// upstream does. The main isolate consumes progress the way a Dart binding
// would: it asks for the latest sequence before the call, reads on a timer,
// and reads once more when the call has returned.
//
//   consumer batch   reads every report since the last one it holds
//   consumer latest  control: looks only at the newest report on each poll,
//                    which is all a single-report API could offer
//
// Exits 0 when the consumer saw exactly the reports the call made, in order,
// in two consecutive calls; 6 when it did not.
import 'dart:async';
import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';

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

const int batchSize = 64;

void worker((SendPort, String, List<int>) arguments) {
  final (replies, libraryPath, [images, steps, stepUs, tiles, tileUs]) =
      arguments;
  final library = DynamicLibrary.open(libraryPath);
  final load = library
      .lookupFunction<Long Function(Int, Int), int Function(int, int)>(
        'stress_report_progress',
      );
  final generate = library
      .lookupFunction<
        Void Function(Int, Int, Int, Int, Int),
        void Function(int, int, int, int, int)
      >('stress_generate_images');
  // A load reports too; the consumer must not take it for the generation.
  load(20, 200);
  final commands = ReceivePort();
  replies.send(commands.sendPort);
  commands.listen((_) {
    generate(images, steps, stepUs, tiles, tileUs);
    replies.send('generated');
  });
}

List<String> reportsOfOneCall(int images, int steps, int tiles) => [
  for (var image = 0; image < images; image++)
    for (var step = 0; step <= steps; step++) '$step/$steps',
  if (tiles > 0)
    for (var image = 0; image < images; image++) ...[
      for (var tile = 0; tile <= tiles; tile++) '$tile/$tiles',
      '$tiles/$tiles',
    ],
];

Future<void> main(List<String> arguments) async {
  final libraryPath = arguments[0];
  final consumer = arguments[1];
  final numbers = [for (final text in arguments.skip(2)) int.parse(text)];
  final [images, steps, _, tiles, _, pollMs] = numbers;

  final library = DynamicLibrary.open(libraryPath);
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

  final replies = ReceivePort();
  final inbox = StreamIterator<Object?>(replies);
  await Isolate.spawn(worker, (
    replies.sendPort,
    libraryPath,
    numbers.sublist(0, 5),
  ));
  await inbox.moveNext();
  final commands = inbox.current as SendPort;

  final expected = reportsOfOneCall(images, steps, tiles);
  for (var call = 0; call < 2; call++) {
    final seen = <String>[];
    read(0, nullptr, 0, latest);
    var after = latest.value;

    void poll() {
      if (consumer == 'latest') {
        read(0, nullptr, 0, latest);
        if (latest.value != after) {
          after = latest.value;
          read(after - 1, reports, 1, latest);
          seen.add('${reports[0].step}/${reports[0].steps}');
        }
        return;
      }
      do {
        final count = read(after, reports, batchSize, latest);
        for (var i = 0; i < count; i++) {
          if (reports[i].sequence != after + 1) {
            stderr.writeln('images: report ${after + 1} is missing');
            exit(6);
          }
          after = reports[i].sequence;
          seen.add('${reports[i].step}/${reports[i].steps}');
        }
      } while (after < latest.value);
    }

    final timer = Timer.periodic(Duration(milliseconds: pollMs), (_) => poll());
    commands.send(null);
    await inbox.moveNext();
    poll();
    timer.cancel();

    if (seen.join(' ') != expected.join(' ')) {
      stderr.writeln('images: call $call reported ${expected.length} times:');
      stderr.writeln('  ${expected.join(' ')}');
      stderr.writeln('images: the $consumer consumer saw ${seen.length}:');
      stderr.writeln('  ${seen.join(' ')}');
      exit(6);
    }
  }
  stdout.writeln(
    'images: the $consumer consumer saw all ${expected.length} reports of '
    'each call',
  );
  exit(0);
}

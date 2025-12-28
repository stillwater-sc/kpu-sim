#include <sw/trace/trace_logger.hpp>
#include <sw/trace/resource_tracker.hpp>
#include <sw/trace/concurrency_analyzer.hpp>
#include <iostream>
#include <vector>

using namespace sw::trace;

int main() {
   std::cout << "1. Testing TraceLogger..." << std::endl;
   TraceLogger& logger = TraceLogger::instance();
   logger.clear();
   std::cout << "   Cleared." << std::endl;

   TraceLogger::Config config;
   config.enabled = true;
   config.buffer_reserve = 1000;
   logger.initialize(config);
   std::cout << "   Initialized." << std::endl;

   std::cout << "2. Testing ResourceTracker..." << std::endl;
   ResourceTracker tracker;
   tracker.register_resource(ComponentType::SYSTOLIC_ARRAY, 0);
   tracker.register_resource(ComponentType::DMA_ENGINE, 0);
   std::cout << "   Registered resources." << std::endl;

   std::cout << "3. Testing simple computation..." << std::endl;
   std::vector<float> data(1000000, 1.0f);
   float sum = 0.0f;
   for (const auto& v : data) sum += v;
   std::cout << "   Sum: " << sum << std::endl;

   std::cout << "Done!" << std::endl;
   return 0;
}
